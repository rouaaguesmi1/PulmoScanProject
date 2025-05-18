<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Validator as ValidatorFacade; // Alias for Validator
use Illuminate\Validation\Rules\File as FileValidationRule;

class NoduleClassificationController extends Controller
{
    /**
     * Display the nodule classification form.
     *
     * @return \Illuminate\View\View
     */
    public function showClassificationForm()
    {
        return view('nodule_classification.form');
    }

    /**
     * Handle the nodule classification request.
     *
     * @param  \Illuminate\Http\Request  $request
     * @return \Illuminate\Http\JsonResponse
     */
    public function processNoduleClassification(Request $request)
    {
        $validator = ValidatorFacade::make($request->all(), [
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
                FileValidationRule::default()->max(512 * 1024), // Max 512MB for RAW
            ],
            'candidates_json_str' => [
                'required',
                'string',
                'json', // Validates if the string is valid JSON
                // Add a custom rule to check if it's an array of objects with required keys if needed
                function ($attribute, $value, $fail) {
                    $data = json_decode($value, true);
                    if (!is_array($data)) {
                        $fail($attribute . ' must be a valid JSON array string.');
                        return;
                    }
                    if (empty($data) && $value !== '[]') { // Allow empty array string '[]'
                        $fail($attribute . ' cannot be an empty JSON array unless it is "[]".');
                        return;
                    }
                    foreach ($data as $index => $item) {
                        if (!is_array($item)) {
                            $fail("Item at index {$index} in " . $attribute . ' is not a valid object.');
                            return;
                        }
                        if (!isset($item['id']) || !isset($item['coordX']) || !isset($item['coordY']) || !isset($item['coordZ'])) {
                            $fail("Item at index {$index} in " . $attribute . ' is missing one or more required keys (id, coordX, coordY, coordZ).');
                            return;
                        }
                        if (!is_numeric($item['coordX']) || !is_numeric($item['coordY']) || !is_numeric($item['coordZ'])) {
                            $fail("Coordinates (coordX, coordY, coordZ) for item with id '{$item['id']}' in " . $attribute . ' must be numeric.');
                            return;
                        }
                    }
                },
            ],
        ]);

        if ($validator->fails()) {
            return response()->json([
                'success' => false,
                'message' => 'Validation failed.',
                'errors' => $validator->errors()
            ], 422);
        }

        $fastApiServiceUrl = rtrim(config('services.fastapi.base_url'), '/');
        $apiEndpoint = $fastApiServiceUrl . '/api/nodule-classification/classify_candidates';

        try {
            $mhdFile = $request->file('mhd_file');
            $rawFile = $request->file('raw_file');
            $candidatesJsonStr = $request->input('candidates_json_str');

            // Note: Http::attach also accepts simple key-value pairs for other form fields
            // So, candidates_json_str will be sent as a regular form field.
            $response = Http::timeout(300) // Long timeout
                ->attach('mhd_file', file_get_contents($mhdFile->path()), $mhdFile->getClientOriginalName())
                ->attach('raw_file', file_get_contents($rawFile->path()), $rawFile->getClientOriginalName())
                ->post($apiEndpoint, [
                    'candidates_json_str' => $candidatesJsonStr, // Sent as a form field
                ]);

            if ($response->successful()) {
                $results = $response->json();
                return response()->json([
                    'success' => true,
                    'classification_results' => $results, // Key for JS
                    'success_message' => $results['message'] ?? 'Candidates classified successfully.'
                ]);
            } else {
                $errorDetails = $response->json();
                $errorMessage = 'API Error: ';
                 if (isset($errorDetails['detail'])) {
                    if (is_array($errorDetails['detail'])) {
                        $validationMessages = [];
                        foreach ($errorDetails['detail'] as $err) {
                             if (isset($err['loc']) && is_array($err['loc']) && count($err['loc']) > 1) {
                                $paramName = $err['loc'][count($err['loc']) -1];
                                // Check if it's a nested validation error within candidates_json_str
                                if (is_string($paramName) && strpos($paramName, 'candidates_json_str') !== false && isset($err['ctx']['error'])) {
                                     $validationMessages[] = "Field 'candidates_json_str': " . $err['ctx']['error'];
                                } else {
                                     $validationMessages[] = "Field '" . $paramName . "': " . $err['msg'];
                                }
                            } else {
                                $validationMessages[] = $err['msg'];
                            }
                        }
                        $errorMessage .= implode('; ', $validationMessages);
                    } else {
                        $errorMessage .= $errorDetails['detail'];
                    }
                } elseif (isset($errorDetails['message'])) {
                    $errorMessage .= $errorDetails['message'];
                } else {
                    $errorMessage .= ('Unknown error from classification service. Status: ' . $response->status());
                }

                Log::error('FastAPI Nodule Classification Error:', ['status' => $response->status(), 'body' => $errorDetails]);
                return response()->json([
                    'success' => false,
                    'api_error' => $errorMessage
                ], $response->status());
            }

        } catch (\Illuminate\Http\Client\ConnectionException $e) {
            Log::error('Connection Exception (Nodule Classification): ' . $e->getMessage());
            return response()->json([
                'success' => false,
                'api_error' => 'Could not connect to the nodule classification service.'
            ], 503);
        } catch (\Exception $e) {
            Log::error('General Exception (Nodule Classification): ' . $e->getMessage(), ['trace' => substr($e->getTraceAsString(),0, 2000)]);
            return response()->json([
                'success' => false,
                'api_error' => 'An unexpected server error occurred: ' . $e->getMessage()
            ], 500);
        }
    }
}