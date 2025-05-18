<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;
use Illuminate\Validation\Rules\File; // For more specific file validation

class ImageEnhancementController extends Controller
{
    /**
     * Display the image enhancement form.
     *
     * @return \Illuminate\View\View
     */
    public function showEnhancementForm()
    {
        return view('image_enhancement.form'); // Path to your blade file
    }

    /**
     * Handle the image enhancement request.
     *
     * @param  \Illuminate\Http\Request  $request
     * @return \Illuminate\Http\RedirectResponse
     */
    public function processImageEnhancement(Request $request)
    {
        $request->validate([
                'image_file' => [
                    'required',
                    File::image()->max(5 * 1024), // ✅ Valid image file
                    'mimes:png,jpg,jpeg,bmp,tiff', // ✅ MIME type check as separate string rule
                ],
                'contrast_factor' => 'nullable|numeric|min:0|max:10',
            ]);

        $fastApiServiceUrl = rtrim(config('services.fastapi.base_url'), '/');
        $apiEndpoint = $fastApiServiceUrl . '/api/image-enhancement/adjust_contrast/';

        try {
            $imageFile = $request->file('image_file');
            $contrastFactor = $request->input('contrast_factor', 2.0); // Default to 2.0 if not provided

            // Prepare multipart data
            // FastAPI expects the file under the key 'file'
            $response = Http::timeout(60) // Increased timeout for potentially larger images/processing
                ->attach(
                    'file', // This MUST match the FastAPI parameter name for the file
                    file_get_contents($imageFile->path()),
                    $imageFile->getClientOriginalName()
                )
                ->post($apiEndpoint, [
                    'contrast_factor' => (float) $contrastFactor,
                ]);

            // Check if the response is successful and an image
            if ($response->successful() && Str::startsWith($response->header('Content-Type'), 'image/')) {
                // Save the enhanced image temporarily
                $enhancedImageContents = $response->body();
                $tempImageDirectory = 'enhanced_images';
                $tempImageName = Str::uuid() . '.png';

                Storage::disk('public')->put($tempImageDirectory . '/' . $tempImageName, $enhancedImageContents);
                $imageUrl = Storage::url($tempImageDirectory . '/' . $tempImageName);// This will be like /storage/enhanced_images/...

                return redirect()->back()
                    ->with('enhanced_image_url', $imageUrl)
                    ->with('success_message', 'Image enhanced successfully.')
                    ->withInput($request->except('image_file'));

            } elseif ($response->clientError() || $response->serverError()) {
                // Handle API error (e.g., FastAPI returned JSON error)
                $errorDetails = $response->json(); // FastAPI ContrastAdjustmentResponse
                $errorMessage = 'API Error: ' . ($errorDetails['message'] ?? 'Unknown error from enhancement service.');
                if (isset($errorDetails['detail'])) { // FastAPI validation errors
                     if (is_array($errorDetails['detail'])) {
                        $validationMessages = [];
                        foreach ($errorDetails['detail'] as $err) {
                            $validationMessages[] = ($err['loc'][1] ?? '') . ': ' . $err['msg'];
                        }
                        $errorMessage = 'API Validation Error: ' . implode(', ', $validationMessages);
                    } else {
                        $errorMessage = 'API Error Detail: ' . $errorDetails['detail'];
                    }
                }
                Log::error('FastAPI Image Enhancement Error:', ['status' => $response->status(), 'body' => $errorDetails]);
                return redirect()->back()
                    ->with('api_error', $errorMessage)
                    ->withInput();
            } else {
                // Unexpected response type
                Log::error('FastAPI Image Enhancement Unexpected Response:', ['status' => $response->status(), 'headers' => $response->headers(), 'body' => substr($response->body(), 0, 500)]);
                return redirect()->back()
                    ->with('api_error', 'Unexpected response from the enhancement service. Content-Type: ' . $response->header('Content-Type'))
                    ->withInput();
            }

        } catch (\Illuminate\Http\Client\ConnectionException $e) {
            Log::error('Connection Exception (Image Enhancement): ' . $e->getMessage());
            return redirect()->back()
                ->with('api_error', 'Could not connect to the image enhancement service. Please try again later.')
                ->withInput();
        } catch (\Exception $e) {
            Log::error('General Exception (Image Enhancement): ' . $e->getMessage(), ['trace' => $e->getTraceAsString()]);
            return redirect()->back()
                ->with('api_error', 'An unexpected error occurred: ' . $e->getMessage())
                ->withInput();
        }
    }
}