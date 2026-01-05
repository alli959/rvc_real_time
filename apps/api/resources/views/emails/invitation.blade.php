<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>You're Invited to MorphVox</title>
</head>
<body style="margin: 0; padding: 0; background-color: #111827; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color: #111827;">
        <tr>
            <td align="center" style="padding: 40px 20px;">
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="max-width: 600px; background-color: #1f2937; border-radius: 12px; overflow: hidden;">
                    <!-- Header -->
                    <tr>
                        <td style="padding: 40px 40px 30px; text-align: center; background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);">
                            <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: 700;">MorphVox</h1>
                            <p style="margin: 10px 0 0; color: rgba(255,255,255,0.8); font-size: 14px;">Voice Transformation Platform</p>
                        </td>
                    </tr>

                    <!-- Content -->
                    <tr>
                        <td style="padding: 40px;">
                            <h2 style="margin: 0 0 20px; color: #ffffff; font-size: 22px; font-weight: 600;">You've been invited!</h2>
                            
                            <p style="margin: 0 0 20px; color: #9ca3af; font-size: 15px; line-height: 1.6;">
                                <strong style="color: #ffffff;">{{ $invitedBy }}</strong> has invited you to join MorphVox — a powerful voice transformation platform.
                            </p>

                            @if(!empty($personalMessage))
                            <div style="margin: 0 0 30px; padding: 20px; background-color: #374151; border-radius: 8px; border-left: 4px solid #7c3aed;">
                                <p style="margin: 0 0 8px; color: #9ca3af; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Personal Message</p>
                                <p style="margin: 0; color: #e5e7eb; font-size: 14px; line-height: 1.5; font-style: italic;">"{{ $personalMessage }}"</p>
                            </div>
                            @endif

                            <p style="margin: 0 0 30px; color: #9ca3af; font-size: 15px; line-height: 1.6;">
                                Click the button below to create your account and get started:
                            </p>

                            <!-- CTA Button -->
                            <table role="presentation" cellspacing="0" cellpadding="0" style="margin: 0 auto 30px;">
                                <tr>
                                    <td style="border-radius: 8px; background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);">
                                        <a href="{{ $inviteUrl }}" target="_blank" style="display: inline-block; padding: 16px 32px; color: #ffffff; font-size: 16px; font-weight: 600; text-decoration: none;">
                                            Accept Invitation →
                                        </a>
                                    </td>
                                </tr>
                            </table>

                            <p style="margin: 0 0 10px; color: #6b7280; font-size: 13px; text-align: center;">
                                This invitation link will expire in 7 days.
                            </p>

                            <p style="margin: 0; color: #6b7280; font-size: 12px; text-align: center; word-break: break-all;">
                                If the button doesn't work, copy and paste this URL:<br>
                                <a href="{{ $inviteUrl }}" style="color: #7c3aed;">{{ $inviteUrl }}</a>
                            </p>
                        </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                        <td style="padding: 30px 40px; background-color: #111827; border-top: 1px solid #374151;">
                            <p style="margin: 0; color: #6b7280; font-size: 12px; text-align: center;">
                                You received this email because someone invited you to MorphVox.<br>
                                If you didn't expect this invitation, you can safely ignore this email.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
