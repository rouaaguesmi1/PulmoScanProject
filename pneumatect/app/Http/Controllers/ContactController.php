<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Mail;

class ContactController extends Controller
{
    public function send(Request $request)
    {
        // Validate the request
        $request->validate([
            'name'    => 'required|string|max:255',
            'email'   => 'required|email',
            'subject' => 'nullable|string|max:255',
            'message' => 'required|string',
        ]);

        // You can either email the message, save it to the database, or both
        // Here's an example of sending an email (you need to configure mail in .env)

        Mail::raw("Message from: {$request->name} <{$request->email}>\n\nSubject: {$request->subject}\n\n{$request->message}", function ($mail) use ($request) {
            $mail->to('mghorbali3@gmail.com')
                ->subject('New Contact Message: ' . ($request->subject ?: 'No Subject'));
        });

        return back()->with('success', 'Your message has been sent successfully!');
    }
}