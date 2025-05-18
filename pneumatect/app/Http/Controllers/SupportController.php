<?php

namespace App\Http\Controllers;
use Illuminate\Support\Facades\Log; 
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Mail; // If you plan to send email
use Illuminate\View\View;
// Potentially use a Mailable: use App\Mail\SupportTicketSubmitted;

class SupportController extends Controller
{
    /**
     * Display the contact support form.
     */
    public function create(): View
    {
        // You can pass categories or other data if needed
        $supportCategories = [
            'technical_issue' => 'Technical Issue',
            'billing_question' => 'Billing Question',
            'feature_request' => 'Feature Request',
            'general_feedback' => 'General Feedback',
            'other' => 'Other',
        ];
        return view('support.contact', ['categories' => $supportCategories]);
    }

    /**
     * Handle the submission of the contact support form.
     */
    public function store(Request $request): RedirectResponse
    {
        $validatedData = $request->validate([
            'category' => ['required', 'string', 'max:255'],
            'subject' => ['required', 'string', 'max:255'],
            'message' => ['required', 'string', 'max:5000'],
            'attachment' => ['nullable', 'file', 'mimes:jpg,jpeg,png,pdf,zip', 'max:5120'], // Optional attachment, max 5MB
        ]);

        $user = Auth::user();

        // ** Placeholder for actual logic **
        // For now, we'll just log it and redirect back with a success message.
        // In a real application, you would:
        // 1. Store the ticket in a database.
        // 2. Send an email notification to your support team (e.g., using a Mailable).
        // 3. Send a confirmation email to the user.
        // 4. Handle file attachment storage if provided.

        $filePath = null;
        if ($request->hasFile('attachment')) {
            $file = $request->file('attachment');
            // Example: store in 'support_attachments/user_id/unique_name.ext'
            $filePath = $file->store('support_attachments/' . $user->id, 'private'); // Use 'private' disk or configure a specific one
        }

        Log::info("Support Ticket Submitted by User ID: {$user->id}", [
            'category' => $validatedData['category'],
            'subject' => $validatedData['subject'],
            'message' => $validatedData['message'],
            'attachment_path' => $filePath,
        ]);
        
        // Example: Send an email (you'd need to create the Mailable)
        // Mail::to('support@yourdomain.com')->send(new SupportTicketSubmitted($user, $validatedData, $filePath));

        return redirect()->route('contact-support.create')
                         ->with('success', 'Your support request has been submitted! We will get back to you shortly.');
    }
}