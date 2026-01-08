<?php

namespace Database\Seeders;

use App\Models\VoiceModel;
use Illuminate\Database\Seeder;
use Illuminate\Support\Str;

class VoiceModelSeeder extends Seeder
{
    /**
     * Seed voice models from local storage
     */
    public function run(): void
    {
        // Define the available models with their local paths
        $models = [
            [
                'name' => 'Bill Cipher',
                'description' => 'Voice of Bill Cipher from Gravity Falls',
                'engine' => 'rvc',
                'model_path' => 'BillCipher/BillCipher.pth',
                'index_path' => 'BillCipher/BillCipher.index',
                'tags' => ['character', 'cartoon', 'gravity-falls', 'villain'],
            ],
            [
                'name' => 'Donald Trump',
                'description' => 'Voice clone of Donald Trump',
                'engine' => 'rvc',
                'model_path' => 'Donald-Trump/Donald-Trump.pth',
                'index_path' => 'Donald-Trump/added_IVF1052_Flat_nprobe_1_Donald-Trump_v2.index',
                'tags' => ['celebrity', 'politician', 'male'],
            ],
            [
                'name' => 'Eminem',
                'description' => 'Voice clone of Eminem',
                'engine' => 'rvc',
                'model_path' => 'EminemTES/EminemTES.pth',
                'index_path' => 'EminemTES/added_IVF2219_Flat_nprobe_1_EminemTES_v2.index',
                'tags' => ['celebrity', 'rapper', 'male', 'music'],
            ],
            [
                'name' => 'Female',
                'description' => 'Generic female voice',
                'engine' => 'rvc',
                'model_path' => 'Female/Female.pth',
                'index_path' => null,
                'tags' => ['female', 'generic'],
            ],
            [
                'name' => 'Homer Simpson',
                'description' => 'Voice of Homer Simpson from The Simpsons',
                'engine' => 'rvc',
                'model_path' => 'Homer Simpson/homer-simpson.pth',
                'index_path' => 'Homer Simpson/added_IVF1568_Flat_nprobe_1_homer-simpson_v2.index',
                'tags' => ['character', 'cartoon', 'simpsons', 'male'],
            ],
            [
                'name' => 'Lois Griffin',
                'description' => 'Voice of Lois Griffin from Family Guy',
                'engine' => 'rvc',
                'model_path' => 'LoisGriffin/LoisGriffin.pth',
                'index_path' => 'LoisGriffin/added_IVF2103_Flat_nprobe_1_LoisGriffin_v2.index',
                'tags' => ['character', 'cartoon', 'family-guy', 'female'],
            ],
            [
                'name' => 'Peter Griffin',
                'description' => 'Voice of Peter Griffin from Family Guy',
                'engine' => 'rvc',
                'model_path' => 'Peter_Griffin/Peter_Griffin.pth',
                'index_path' => 'Peter_Griffin/added_IVF1759_Flat_nprobe_1_Peter_Griffin_v2.index',
                'tags' => ['character', 'cartoon', 'family-guy', 'male'],
            ],
            [
                'name' => 'Spongebob',
                'description' => 'Voice of SpongeBob SquarePants',
                'engine' => 'rvc',
                'model_path' => 'Spongebob/Spongebob.pth',
                'index_path' => 'Spongebob/added_IVF1779_Flat_nprobe_1_Spongebob_v2.index',
                'tags' => ['character', 'cartoon', 'spongebob', 'male'],
            ],
            [
                'name' => 'E-Girl',
                'description' => 'E-girl style voice',
                'engine' => 'rvc',
                'model_path' => 'egirl/egirl.pth',
                'index_path' => null,
                'tags' => ['female', 'internet', 'style'],
            ],
            [
                'name' => 'Gusti',
                'description' => 'Custom Gusti voice model',
                'engine' => 'rvc',
                'model_path' => 'gusti/gusti.pth',
                'index_path' => 'gusti/added_IVF673_Flat_nprobe_1_gusti_v2.index',
                'tags' => ['custom', 'male'],
            ],
            [
                'name' => 'Sigurgeir',
                'description' => 'Custom Sigurgeir voice model (v0.5)',
                'engine' => 'rvc',
                'model_path' => 'sigurgeir-0.5-model/G_1360.pth',
                'index_path' => 'sigurgeir-0.5-model/added_IVF673_Flat_nprobe_1_sigurgeir-0.5-model_v2.index',
                'tags' => ['custom', 'male', 'icelandic'],
            ],
            [
                'name' => 'Anton',
                'description' => 'Custom Anton voice model (v0.5)',
                'engine' => 'rvc',
                'model_path' => 'anton-0.5-model/G_1360.pth',
                'index_path' => 'anton-0.5-model/added_IVF673_Flat_nprobe_1_anton-0.5-model_v2.index',
                'tags' => ['custom', 'male'],
            ],
            [
                'name' => 'Emil',
                'description' => 'Custom Emil voice model (v0.4)',
                'engine' => 'rvc',
                'model_path' => 'emil-0.4-model/G_1360.pth',
                'index_path' => 'emil-0.4-model/added_IVF673_Flat_nprobe_1_emil-0.4-model_v2.index',
                'tags' => ['custom', 'male'],
            ],
        ];

        foreach ($models as $modelData) {
            // Check if already exists
            $existing = VoiceModel::where('name', $modelData['name'])->first();
            if ($existing) {
                $this->command->info("Model {$modelData['name']} already exists, updating...");
                $existing->update([
                    'model_path' => $modelData['model_path'],
                    'index_path' => $modelData['index_path'],
                    'status' => 'ready',
                    'visibility' => 'public',
                ]);
                continue;
            }

            VoiceModel::create([
                'uuid' => Str::uuid(),
                'name' => $modelData['name'],
                'slug' => Str::slug($modelData['name']),
                'description' => $modelData['description'],
                'engine' => $modelData['engine'],
                'model_path' => $modelData['model_path'],
                'index_path' => $modelData['index_path'],
                'tags' => $modelData['tags'],
                'status' => 'ready',
                'visibility' => 'public',
                'storage_type' => 'local',
                'has_consent' => true,
                'user_id' => null, // System model
            ]);

            $this->command->info("Created model: {$modelData['name']}");
        }
    }
}
