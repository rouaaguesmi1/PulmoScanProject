<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Validator as ValidatorFacade;
use Illuminate\Validation\Rules\File as FileValidationRule;

class PatientCancerClassificationController extends Controller
{
    /**
     * Display the patient cancer classification form.
     */
    public function showPredictionForm()
    {
        return view('patient_cancer_classification.form');
    }

    /**
     * Handle the patient cancer classification request.
     */
    public function processPatientPrediction(Request $request)
    {
        $validator = ValidatorFacade::make($request->all(), [
            'scan_zip' => [
                'required',
                'file',
                // Simple check for zip extension
                function ($attribute, $value, $fail) {
                    if (strtolower($value->getClientOriginalExtension()) !== 'zip') {
                        $fail('The ' . $attribute . ' must be a file of type: zip.');
                    }
                },
                FileValidationRule::default()->max(102400), // Max 10GB for ZIP, adjust significantly as needed
            ],
            'patient_id_form' => 'nullable|string|max:100',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'success' => false,
                'message' => 'Validation failed.',
                'errors' => $validator->errors()
            ], 422);
        }

        $fastApiServiceUrl = rtrim(config('services.fastapi.base_url'), '/');
        $apiEndpoint = $fastApiServiceUrl . '/api/patient-cancer-classification/predict_patient_cancer_status';

        try {
            $scanZipFile = $request->file('scan_zip');
            $patientIdForm = $request->input('patient_id_form'); // Can be null

            $httpRequest = Http::timeout(600); // 10 minutes timeout for large ZIP processing

            // Build multipart data
            $multipartData = [
                [
                    'name'     => 'scan_zip',
                    'contents' => fopen($scanZipFile->path(), 'r'),
                    'filename' => $scanZipFile->getClientOriginalName()
                ]
            ];

            if ($patientIdForm !== null) {
                $multipartData[] = [
                    'name'     => 'patient_id_form',
                    'contents' => $patientIdForm
                ];
            }
            
            // Make the request using Http::asMultipart() and post()
            $response = $httpRequest->asMultipart()->post($apiEndpoint, $multipartData);


            if ($response->successful()) {
                $results = $response->json();
                return response()->json([
                    'success' => true,
                    'prediction_results' => $results, // Key for JS
                    'success_message' => $results['message'] ?? 'Patient cancer status predicted successfully.'
                ]);
            } else {
                $errorDetails = $response->json();
                $errorMessage = 'API Error: ';
                if (isset($errorDetails['detail'])) {
                     if (is_array($errorDetails['detail'])) {
                        $validationMessages = [];
                        foreach ($errorDetails['detail'] as $err) {
                             if (isset($err['loc']) && is_array($err['loc']) && count($err['loc']) > 1) {
                                $validationMessages[] = "Field '" . $err['loc'][count($err['loc']) -1] . "': " . $err['msg'];
                            } else {
                                $validationMessages[] = $err['msg'];
                            }
                        }
                        $errorMessage .= implode('; ', $validationMessages);
                    } else {
                        $errorMessage .= $errorDetails['detail'];
                    }
                } elseif (isset($errorDetails['error_details'])) { // Check for the specific error_details key from FastAPI response
                    $errorMessage .= $errorDetails['error_details'];
                } elseif (isset($errorDetails['message'])) {
                    $errorMessage .= $errorDetails['message'];
                } else {
                    $errorMessage .= ('Unknown error from patient classification service. Status: ' . $response->status());
                }


                Log::error('FastAPI Patient Cancer Classification Error:', ['status' => $response->status(), 'body' => $errorDetails]);
                return response()->json([
                    'success' => false,
                    'api_error' => $errorMessage
                ], $response->status()); // Use original error status
            }

        } catch (\Illuminate\Http\Client\ConnectionException $e) {
            Log::error('Connection Exception (Patient Cancer Classification): ' . $e->getMessage());
            return response()->json([
                'success' => false,
                'api_error' => 'Could not connect to the patient classification service.'
            ], 503); // Service Unavailable
        } catch (\Exception $e) {
            Log::error('General Exception (Patient Cancer Classification): ' . $e->getMessage(), ['trace' => substr($e->getTraceAsString(), 0, 2000)]);
            return response()->json([
                'success' => false,
                'api_error' => 'An unexpected server error occurred: ' . $e->getMessage()
            ], 500);
        }
    }
}