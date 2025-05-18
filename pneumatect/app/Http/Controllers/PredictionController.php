<?php

namespace App\Http\Controllers;

use App\Models\SubtypePrediction; // Import the model

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Http; // Import Laravel HTTP Client
use Illuminate\Support\Facades\Log; // Import Log facade
use Illuminate\Support\Facades\Storage; // Import Storage facade
use Illuminate\Support\Str; // Import Str facade
use Illuminate\View\View;
use Illuminate\Http\RedirectResponse; // Import RedirectResponse
use App\Models\LungCancerRiskAssessment; // Import the LungCancerRiskAssessment model
use Barryvdh\DomPDF\Facade\Pdf;
use Carbon\Carbon; // Import Carbon for date and time handling
use App\Models\PatientSurvivalPrediction; // Import the PatientSurvivalPrediction model
use Symfony\Component\HttpFoundation\Response; // For PDF response type hint


class PredictionController extends Controller
{
    public function subtypePredictionCreate(): View
    {
        return view('predictions.subtype_create');
    }

    /**
     * Store the uploaded image, get prediction, and save results.
     */
    public function storeSubtypePrediction(Request $request): RedirectResponse
    {
        // 1. Validate the incoming request
        $request->validate([
            'ct_scan_image' => ['required', 'image', 'mimes:png,jpg,jpeg,dcm', 'max:10240'], // Example validation: required, image, specific types, max 10MB
        ],[
            'ct_scan_image.required' => 'Please select or drop a CT scan image.',
            'ct_scan_image.image' => 'The uploaded file must be an image.',
            'ct_scan_image.mimes' => 'Only PNG, JPG, JPEG, and DICOM (.dcm) files are allowed.',
            'ct_scan_image.max' => 'The image may not be greater than 10MB.',
        ]);

        // 2. Store the image
        $file = $request->file('ct_scan_image');
        $originalName = $file->getClientOriginalName();
        // Store in 'public/ct_scans' using a unique hash name
        $path = $file->store('ct_scans', 'public'); 
        // Or: $path = Storage::disk('s3')->putFile('ct_scans', $file); // For S3

        if (!$path) {
             return back()->with('error', 'Failed to store uploaded image.');
        }

        // 3. Call FastAPI Endpoint
        // IMPORTANT: Update URL if needed for this specific model
        $fastApiUrl = "http://127.0.0.1:8001/api/cancer-subtype/"; 
        $predictionData = null;
        $errorMessage = null;

        try {
            $response = Http::timeout(60) // Set timeout (e.g., 60 seconds)
                ->attach(
                    'file', // Matches the key FastAPI expects
                    Storage::disk('public')->get($path), // Get file content
                    $originalName // Send original filename
                )
                ->post($fastApiUrl); // Add ->acceptJson() if needed

            if ($response->successful()) {
                $predictionData = $response->json();
            } else {
                Log::error("FastAPI Error: Status " . $response->status() . " Body: " . $response->body());
                $errorMessage = "Prediction service failed (Status: " . $response->status() . "). Please try again later.";
            }
        } catch (\Illuminate\Http\Client\ConnectionException $e) {
             Log::error("FastAPI Connection Error: " . $e->getMessage());
             $errorMessage = "Could not connect to the prediction service. Please try again later.";
        } catch (\Exception $e) {
             Log::error("Prediction Request Error: " . $e->getMessage());
             $errorMessage = "An unexpected error occurred during prediction.";
        }

        // 4. Store Prediction Result in Database
        if ($predictionData && !$errorMessage) {
            try {
                SubtypePrediction::create([
                    'user_id' => Auth::id(),
                    'image_path' => $path,
                    'original_filename' => $originalName,
                    'predicted_class' => $predictionData['predicted_class'] ?? null, // Use null coalescing
                    'confidence' => isset($predictionData['confidence']) ? (float)$predictionData['confidence'] : null,
                    'prediction_details' => $predictionData, // Store full response if desired
                ]);

                // Flash prediction result to session for display on redirect
                return redirect()->route('predict.subtype.create')
                       ->with('predictionResult', $predictionData)
                       ->with('success', 'Prediction successful!');

            } catch (\Exception $e) {
                Log::error("Database Save Error: " . $e->getMessage());
                // Optionally delete the stored image if DB save fails
                // Storage::disk('public')->delete($path); 
                $errorMessage = "Prediction successful, but failed to save results.";
                 // Still redirect back, but show the error
                 return redirect()->route('predict.subtype.create')
                        ->with('predictionResult', $predictionData) // Show result anyway
                        ->with('error', $errorMessage); 
            }

        } else {
            // Handle case where prediction failed
            // Optionally delete the stored image if prediction fails
             Storage::disk('public')->delete($path); 
             return back()->with('error', $errorMessage ?? 'Prediction failed for an unknown reason.');
        }
    }



    public function medicalHistoryIndex(): View
    {
        $user = Auth::user();

        // Fetch CT Scan Subtype Predictions
        $subtypePredictions = SubtypePrediction::where('user_id', $user->id)
            ->orderBy('created_at', 'desc')
            ->get();

        // Fetch latest Lung Cancer Risk Assessment
        $latestRiskAssessment = LungCancerRiskAssessment::where('user_id', $user->id)
            ->orderBy('assessment_date', 'desc') // Or created_at
            ->first();

        // Fetch latest Patient Survival Prediction (which now includes prediction results)
        $latestSurvivalPrediction = PatientSurvivalPrediction::where('user_id', $user->id)
            ->orderBy('created_at', 'desc')
            ->first();

        // Helper arrays for displaying gender if needed (assuming survival input might have '0' or '1')
        $genderOptions = ['1' => 'Male', '0' => 'Female'];


        return view('medical_history.index', [
            'user' => $user,
            'subtypePredictions' => $subtypePredictions,
            'latestRiskAssessment' => $latestRiskAssessment,
            'latestSurvivalPrediction' => $latestSurvivalPrediction, // Renamed from latestSurvivalInput for clarity
            'genderOptions' => $genderOptions, // Pass for display if gender is stored as code
        ]);
    }

    /**
     * Generate a consolidated PDF medical history report for the user.
     */
    public function generatePdf(): Response
    {
        $user = Auth::user();

        // Fetch all data needed for the PDF
        $subtypePredictions = SubtypePrediction::where('user_id', $user->id)
            ->orderBy('created_at', 'desc')
            ->get();

        $riskAssessments = LungCancerRiskAssessment::where('user_id', $user->id)
            ->orderBy('assessment_date', 'desc') // Or created_at
            ->get();
        $latestRiskAssessment = $riskAssessments->first(); // Still useful for summary

        $survivalPredictions = PatientSurvivalPrediction::where('user_id', $user->id)
            ->orderBy('created_at', 'desc')
            ->get();
        $latestSurvivalPrediction = $survivalPredictions->first(); // Still useful for summary

        // Helper arrays for display values in PDF
        $genderOptions = ['1' => 'Male', '0' => 'Female']; // Assuming '1' is Male, '0' is Female
        $yesNoOptions = [1 => 'Yes', 0 => 'No', true => 'Yes', false => 'No']; // Handle boolean and int

        // If cancer_stage is stored as an INT in PatientSurvivalPrediction, you'll need a mapping.
        // If it's stored as a string (e.g., "Stage I"), then direct display is fine.
        // Example mapping if DB stored integer:
        // $cancerStageDisplayMapping = [
        //     1 => 'Stage I', 2 => 'Stage IA', /* ... etc ... */ 0 => 'Unknown'
        // ];

        $data = [
            'user' => $user,
            'subtypePredictions' => $subtypePredictions,
            'latestRiskAssessment' => $latestRiskAssessment, // For summary section if needed
            'latestSurvivalPrediction' => $latestSurvivalPrediction, // For summary section if needed
            'riskAssessments' => $riskAssessments, // For detailed history
            'survivalPredictions' => $survivalPredictions, // For detailed history
            'generationDate' => Carbon::now()->format('F j, Y, g:i a'),
            'genderOptions' => $genderOptions,
            'yesNoOptions' => $yesNoOptions,
            // 'cancerStageDisplayMapping' => $cancerStageDisplayMapping, // Pass if needed
        ];

        $pdf = Pdf::loadView('medical_history.pdf', $data);
        // $pdf->setPaper('A4', 'landscape'); // Optional: for wider tables

        return $pdf->download('medical_history_report_' . $user->id . '_' . Carbon::now()->format('YmdHis') . '.pdf');
    }
}