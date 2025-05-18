<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use Illuminate\Validation\Rules\File as FileValidationRule;

class NoduleDetectionController extends Controller
{
public function showDetectionForm()
{
return view('nodule_detection.form');
}

public function processNoduleDetection(Request $request)
{
    $validator = \Illuminate\Support\Facades\Validator::make($request->all(), [
        'mhd_file' => [
            'required',
            'file',
            function ($attribute, $value, $fail) {
                if (strtolower($value->getClientOriginalExtension()) !== 'mhd') {
                    $fail('The ' . $attribute . ' must be a file of type: mhd.');
                }
            },
            FileValidationRule::default()->max(2 * 1024), // Max 2MB for MHD
        ],
        'raw_file' => [
            'required',
            'file',
            function ($attribute, $value, $fail) {
                if (strtolower($value->getClientOriginalExtension()) !== 'raw') {
                    $fail('The ' . $attribute . ' must be a file of type: raw.');
                }
            },
            FileValidationRule::default()->max(512 * 1024), // Max 512MB for RAW, adjust
        ],
    ]);

    if ($validator->fails()) {
        return response()->json([
            'success' => false,
            'message' => 'Validation failed.', // Generic message
            'errors' => $validator->errors()
        ], 422);
    }

    $fastApiServiceUrl = rtrim(config('services.fastapi.base_url'), '/');
    $apiEndpoint = $fastApiServiceUrl . '/api/nodule-detection/predict';

    try {
        $mhdFile = $request->file('mhd_file');
        $rawFile = $request->file('raw_file');

        $response = Http::timeout(300)
            ->attach('mhd_file', file_get_contents($mhdFile->path()), $mhdFile->getClientOriginalName())
            ->attach('raw_file', file_get_contents($rawFile->path()), $rawFile->getClientOriginalName())
            ->post($apiEndpoint);

        if ($response->successful()) {
            $results = $response->json();
            return response()->json([
                'success' => true,
                'nodule_detection_results' => $results,
                'success_message' => $results['message'] ?? 'Nodule detection processed successfully.'
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
                    $errorMessage .= $errorDetails['detail']; // Fallback if detail is not an array of errors
                }
            } elseif (isset($errorDetails['message'])) {
                $errorMessage .= $errorDetails['message'];
            } else {
                $errorMessage .= 'Unknown error from detection service. Status: ' . $response->status();
            }

            Log::error('FastAPI Nodule Detection Error:', ['status' => $response->status(), 'body' => $errorDetails]);
            return response()->json([
                'success' => false,
                'api_error' => $errorMessage
            ], $response->status()); // Use original error status if appropriate
        }

    } catch (\Illuminate\Http\Client\ConnectionException $e) {
        Log::error('Connection Exception (Nodule Detection): ' . $e->getMessage());
        return response()->json([
            'success' => false,
            'api_error' => 'Could not connect to the nodule detection service. Please try again later.'
        ], 503); // Service Unavailable
    } catch (\Exception $e) {
        Log::error('General Exception (Nodule Detection): ' . $e->getMessage(), ['trace' => substr($e->getTraceAsString(), 0, 2000)]);
        return response()->json([
            'success' => false,
            'api_error' => 'An unexpected server error occurred: ' . $e->getMessage()
        ], 500);
    }
}
}