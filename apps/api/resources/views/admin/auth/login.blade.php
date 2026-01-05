<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>MorphVox Admin Login</title>
    <style>
        body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 0; background: #0b1020; color: #e6e6e6; display:flex; min-height: 100vh; align-items:center; justify-content:center; }
        .card { background: #0f1630; border: 1px solid #1f2a4d; border-radius: 12px; padding: 18px; width: 100%; max-width: 420px; }
        input { background: #0b1020; color: #e6e6e6; border: 1px solid #26345e; border-radius: 10px; padding: 10px; width: 100%; box-sizing: border-box; }
        label { display:block; margin-bottom: 6px; font-size: 12px; color: #aab3cc; }
        .btn { display:inline-block; padding: 10px 12px; border-radius: 10px; border: 1px solid #2b5bbb; background: #13224b; color: #e6e6e6; cursor: pointer; width:100%; }
        .btn:hover { background: #173064; }
        .errors { background: #2a0b0b; border: 1px solid #5a1f1f; padding: 10px 12px; border-radius: 12px; margin-bottom: 12px; }
        a { color: #8ab4f8; }
    </style>
</head>
<body>

<div class="card">
    <h2 style="margin: 0 0 6px 0;">MorphVox Admin</h2>
    <div style="margin: 0 0 16px 0; color:#aab3cc; font-size: 13px;">Sign in with an <b>admin</b> account.</div>

    @if ($errors->any())
        <div class="errors">
            <ul style="margin: 0; padding-left: 18px;">
                @foreach ($errors->all() as $error)
                    <li>{{ $error }}</li>
                @endforeach
            </ul>
        </div>
    @endif

    <form method="POST" action="{{ route('admin.login.post') }}">
        @csrf

        <div style="margin-bottom: 12px;">
            <label>Email</label>
            <input type="email" name="email" value="{{ old('email') }}" required>
        </div>

        <div style="margin-bottom: 14px;">
            <label>Password</label>
            <input type="password" name="password" required>
        </div>

        <button class="btn" type="submit">Login</button>
    </form>
</div>

</body>
</html>
