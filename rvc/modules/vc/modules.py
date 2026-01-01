import logging
import os
import traceback
import json
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from rvc.configs.config import Config
try:
    from rvc.lib.audio import load_audio, wav2  # type: ignore
except Exception:  # pragma: no cover
    load_audio = None  # type: ignore
    wav2 = None  # type: ignore

from rvc.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rvc.modules.vc.pipeline import Pipeline
from rvc.modules.vc.utils import *

logger: logging.Logger = logging.getLogger(__name__)


class VC:
    def __init__(self):
        self.n_spk: any = None
        self.tgt_sr: int | None = None
        self.net_g = None
        self.pipeline: Pipeline | None = None
        self.cpt: OrderedDict | None = None
        self.version: str | None = None
        self.if_f0: int | None = None
        self.hubert_model: any = None
        self.config = Config()

    def get_vc(self, sid: str, *to_return_protect: int):
        logger.info("Get sid: " + sid)

        person = sid if os.path.exists(sid) else f'{os.getenv("weight_root")}/{sid}'
        logger.info(f"Loading: {person}")

        # load checkpoint
        self.cpt = torch.load(person, map_location="cpu")

        # normalize checkpoint formats (WebUI vs other forks)
        if isinstance(self.cpt, dict) and "weight" not in self.cpt:
            if "model" in self.cpt and isinstance(self.cpt["model"], dict):
                self.cpt["weight"] = self.cpt["model"]
            elif "state_dict" in self.cpt and isinstance(self.cpt["state_dict"], dict):
                self.cpt["weight"] = self.cpt["state_dict"]
            elif "net_g" in self.cpt and isinstance(self.cpt["net_g"], dict):
                self.cpt["weight"] = self.cpt["net_g"]

        if not isinstance(self.cpt, dict):
            self.cpt = {"weight": self.cpt}

        # ✅ safe to read after load/normalize
        self.if_f0 = int(self.cpt.get("f0", 1) or 1)

        return_protect = [
            to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5,
            to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33,
        ]

        w = self.cpt["weight"]

        emb_key = "emb_g.weight"
        if emb_key not in w:
            emb_key = "module.emb_g.weight"
        n_spk = w[emb_key].shape[0]

        model_dir = Path(person).resolve().parent
        cfg_json = model_dir / "config.json"
        tgt_sr = None

        # Detect version from weights if possible
        k = "enc_p.emb_phone.weight"
        if k not in w:
            k = "module.enc_p.emb_phone.weight"
        if k in w:
            feat_dim = w[k].shape[1]  # 256 (v1) or 768 (v2)
            self.version = "v2" if feat_dim == 768 else "v1"
        else:
            self.version = self.cpt.get("version", "v1")

        # Try to load config from config.json or use defaults
        if cfg_json.exists():
            j = json.loads(cfg_json.read_text(encoding="utf-8"))
            data = j.get("data", {}) if isinstance(j, dict) else {}
            m = j.get("model", {}) if isinstance(j, dict) else {}
            tgt_sr = data.get("sampling_rate") or data.get("sr") or data.get("sample_rate")
            logger.info(f"Found config.json, loading model parameters")
        else:
            # No config.json - use default RVC parameters
            m = {}
            tgt_sr = None
            logger.info(f"No config.json found, using default {self.version} parameters")

        # Build config - use values from config.json if available, otherwise defaults
        spec_channels = 80
        segment_size = 0
        inter_channels = int(m.get("inter_channels", 192))
        hidden_channels = int(m.get("hidden_channels", 192))
        filter_channels = int(m.get("filter_channels", 768))
        n_heads = int(m.get("n_heads", 2))
        n_layers = int(m.get("n_layers", 6))
        kernel_size = int(m.get("kernel_size", 3))
        p_dropout = float(m.get("p_dropout", 0.0) or 0.0)

        resblock = str(m.get("resblock", "1"))
        resblock_kernel_sizes = m.get("resblock_kernel_sizes", [3, 7, 11])
        resblock_dilation_sizes = m.get(
            "resblock_dilation_sizes",
            [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        )

        # Detect sample rate from upsample rates if possible
        # v1/v2 32k: [10, 4, 2, 2, 2] -> 320 hop -> 32000
        # v1/v2 40k: [10, 5, 2, 2, 2] -> 400 hop -> 40000  (or [10, 8, 2, 2])
        # v1/v2 48k: [10, 6, 2, 2, 2] -> 480 hop -> 48000  (or [12, 10, 2, 2])
        default_upsample_rates = m.get("upsample_rates", [10, 6, 2, 2, 2])
        upsample_rates = default_upsample_rates
        upsample_initial_channel = int(m.get("upsample_initial_channel", 512))
        
        # Try to infer sample rate from upsample configuration
        if not tgt_sr:
            hop_length = 1
            for r in upsample_rates:
                hop_length *= r
            # Common RVC sample rates: 32000, 40000, 48000
            tgt_sr = hop_length * 100  # hop_length * 100 gives approximate sr
            # Clamp to known valid rates
            if tgt_sr <= 32000:
                tgt_sr = 32000
            elif tgt_sr <= 40000:
                tgt_sr = 40000
            else:
                tgt_sr = 48000

        # Adjust upsample kernel sizes based on rates
        default_upsample_kernel_sizes = m.get("upsample_kernel_sizes")
        if default_upsample_kernel_sizes:
            upsample_kernel_sizes = default_upsample_kernel_sizes
        else:
            # Default kernel sizes are typically 2x the upsample rate
            upsample_kernel_sizes = [r * 2 for r in upsample_rates]

        gin_channels = int(m.get("gin_channels", 256))
        sr = int(tgt_sr)

        # Only build config if not already present in checkpoint
        if "config" not in self.cpt or not isinstance(self.cpt["config"], list):
            self.cpt["config"] = [
                spec_channels,
                segment_size,
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                resblock,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                n_spk,
                gin_channels,
                sr,
            ]

            logger.info(
                f"Built config (sr={sr}, n_spk={n_spk}, version={self.version})"
            )

        self.tgt_sr = int(self.cpt["config"][-1])

        if not self.version:
            self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        if sid == "" or []:
            logger.info("Clean model cache")
            del (self.hubert_model, self.tgt_sr, self.net_g)
            (self.net_g) = self.n_spk = index = None
        else:
            self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            self.net_g.eval().to(self.config.device)
            self.net_g = self.net_g.half() if self.config.is_half else self.net_g.float()
            self.pipeline = Pipeline(self.tgt_sr, self.config)
            self.n_spk = n_spk
            index = get_index_path_from_model(sid)
            logger.info("Select index: " + index)

        return self.n_spk, return_protect, index

        

    def vc_single(
        self,
        sid: int,
        input_audio_path: Path,
        f0_up_key: int = 0,
        f0_method: str = "rmvpe",
        f0_file: Path | None = None,
        index_file: Path | None = None,
        index_rate: float = 0.75,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        hubert_path: str | None = None,
    ):
        hubert_path = os.getenv("hubert_path") if not hubert_path else hubert_path

        if load_audio is None or wav2 is None:
            raise ImportError(
                "PyAV dependency is missing. Install the 'av' package to use file-based conversion (vc_single/vc_multi). "
                "Realtime conversion in rvc_real_time does not require PyAV."
            )

        try:
            audio = load_audio(input_audio_path, 16000)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max

            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config, hubert_path)

            f0_file = open(f0_file, "r") if f0_file else None
            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                input_audio_path,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                f0_file,
            )
            f0_file.close() if f0_file else None

            if output_path is None:
                return (
                    "Success.",
                    (self.tgt_sr, audio_opt),
                    (self.tgt_sr, audio_opt),
                    None,
                )
            else:
                wav2(output_path, audio_opt, self.tgt_sr)
                return (
                    "Success.",
                    (self.tgt_sr, audio_opt),
                    (self.tgt_sr, audio_opt),
                    None,
                )
        except Exception:
            info = traceback.format_exc()
            logger.warning(info)
            return None, None, None, info

    def vc_multi(
        self,
        sid: int,
        paths: list,
        opt_root: Path,
        f0_up_key: int = 0,
        f0_method: str = "rmvpe",
        f0_file: Path | None = None,
        index_file: Path | None = None,
        index_rate: float = 0.75,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        output_format: str = "wav",
        hubert_path: str | None = None,
    ):
        try:
            os.makedirs(opt_root, exist_ok=True)
            paths = [path.name for path in paths]
            infos = []
            for path in paths:
                tgt_sr, audio_opt, _, info = self.vc_single(
                    sid,
                    Path(path),
                    f0_up_key,
                    f0_method,
                    f0_file,
                    index_file,
                    index_rate,
                    filter_radius,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                    hubert_path,
                )
                if info:
                    try:
                        if output_format in ["wav", "flac"]:
                            sf.write(
                                f"{opt_root}/{os.path.basename(path)}.{output_format}",
                                audio_opt,
                                tgt_sr,
                            )
                        else:
                            with BytesIO() as wavf:
                                sf.write(wavf, audio_opt, tgt_sr, format="wav")
                                wavf.seek(0, 0)
                                with open(
                                    f"{opt_root}/{os.path.basename(path)}.{output_format}",
                                    "wb",
                                ) as outf:
                                    wav2(wavf, outf, output_format)
                    except Exception:
                        info += traceback.format_exc()
                infos.append(f"{os.path.basename(path)}->{info}")
                yield "\n".join(infos)
            yield "\n".join(infos)
        except:
            yield traceback.format_exc()
