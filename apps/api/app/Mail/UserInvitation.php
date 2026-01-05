<?php

namespace App\Mail;

use Illuminate\Bus\Queueable;
use Illuminate\Mail\Mailable;
use Illuminate\Mail\Mailables\Content;
use Illuminate\Mail\Mailables\Envelope;
use Illuminate\Queue\SerializesModels;

class UserInvitation extends Mailable
{
    use Queueable, SerializesModels;

    public string $inviteUrl;
    public ?string $personalMessage;
    public string $invitedBy;

    public function __construct(string $inviteUrl, ?string $personalMessage, string $invitedBy)
    {
        $this->inviteUrl = $inviteUrl;
        $this->personalMessage = $personalMessage;
        $this->invitedBy = $invitedBy;
    }

    public function envelope(): Envelope
    {
        return new Envelope(
            subject: 'You\'ve been invited to MorphVox',
        );
    }

    public function content(): Content
    {
        return new Content(
            view: 'emails.invitation',
            with: [
                'inviteUrl' => $this->inviteUrl,
                'personalMessage' => $this->personalMessage,
                'invitedBy' => $this->invitedBy,
            ],
        );
    }
}
