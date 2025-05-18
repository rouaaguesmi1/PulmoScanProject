<?php

namespace App\Http\Controllers;

use App\Models\PatientSurvivalPrediction;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use Illuminate\View\View;
use Carbon\Carbon;

class PatientSurvivalController extends Controller
{
    public function create(): View
    {
        $user = Auth::user();
        $latestAssessment = PatientSurvivalPrediction::where('user_id', $user->id)
            ->orderBy('created_at', 'desc')
            ->first();

        // Ensure integer keys for gender for form submission if API expects int
        $genderOptions = ['1' => 'Male', '0' => 'Female']; // 1: Male, 0: Female (adjust if API is different)

        $countryOptions = $this->getCountryOptions(); // Assuming this helper exists and is fine

        // Keys for cancerStageOptions are the strings submitted by the form. Mapping to int happens in store().
        $cancerStageOptions = [
            'Stage I' => 'Stage I', 'Stage IA' => 'Stage IA', 'Stage IB' => 'Stage IB',
            'Stage II' => 'Stage II', 'Stage IIA' => 'Stage IIA', 'Stage IIB' => 'Stage IIB',
            'Stage III' => 'Stage III', 'Stage IIIA' => 'Stage IIIA', 'Stage IIIB' => 'Stage IIIB', 'Stage IIIC' => 'Stage IIIC',
            'Stage IV' => 'Stage IV', 'Stage IVA' => 'Stage IVA', 'Stage IVB' => 'Stage IVB',
            'Unknown' => 'Unknown',
        ];
        $yesNoOptions = ['1' => 'Yes', '0' => 'No']; // Values '1' and '0' (strings) are fine, will be cast to int for API

        // Keys for smokingStatusOptions are the strings submitted. One-hot encoding in store().
        $smokingStatusOptions = [
            'Current Smoker' => 'Current Smoker',
            'Former Smoker' => 'Former Smoker',
            'Never Smoked' => 'Never Smoked',
            'Passive Smoker' => 'Passive Smoker',
        ];

        // Keys for treatmentTypeOptions are strings. One-hot encoding/mapping in store().
        $treatmentTypeOptions = [
            'Surgery' => 'Surgery',
            'Chemotherapy' => 'Chemotherapy',
            'Radiation Therapy' => 'Radiation Therapy', // API might expect "Radiation"
            'Chemoradiation' => 'Chemoradiation',       // API might expect "Combined"
            'Targeted Therapy' => 'Targeted Therapy',
            'Immunotherapy' => 'Immunotherapy',
            'Palliative Care' => 'Palliative Care',
            'None' => 'None',
            'Other' => 'Other',
        ];

        return view('predictions.patient_survival_create', [
            'latestAge' => $latestAssessment->age_at_diagnosis ?? null,
            'latestGender' => $latestAssessment->gender ?? null, // This should be '0' or '1' if pre-filled
            'latestDiagnosisDate' => $latestAssessment->diagnosis_date ?? null,
            'latestFamilyHistory' => $latestAssessment->family_history_of_cancer ?? null,
            'latestHypertension' => $latestAssessment->hypertension ?? null,
            'latestAsthma' => $latestAssessment->asthma ?? null,
            'latestCirrhosis' => $latestAssessment->cirrhosis ?? null,
            'latestOtherCancerHistory' => $latestAssessment->other_cancer_history ?? null,
            // Add pre-fill for new date fields if applicable (likely not on first create)
            'latestBeginningOfTreatmentDate' => $latestAssessment->beginning_of_treatment_date ?? null,
            'latestEndOfTreatmentDate' => $latestAssessment->end_of_treatment_date ?? null,


            'genderOptions' => $genderOptions,
            'countryOptions' => $countryOptions,
            'cancerStageOptions' => $cancerStageOptions,
            'yesNoOptions' => $yesNoOptions,
            'smokingStatusOptions' => $smokingStatusOptions,
            'treatmentTypeOptions' => $treatmentTypeOptions,
        ]);
    }

    public function store(Request $request): RedirectResponse
    {
        $user = Auth::user();
        $latestAssessment = PatientSurvivalPrediction::where('user_id', $user->id)
            ->orderBy('created_at', 'desc')
            ->first();

        $ageRule = isset($latestAssessment->age_at_diagnosis) ? ['nullable', 'integer', 'min:1', 'max:120'] : ['required', 'integer', 'min:1', 'max:120'];
        $genderRule = isset($latestAssessment->gender) ? ['nullable', 'in:0,1'] : ['required', 'in:0,1'];
        $diagnosisDateRule = isset($latestAssessment->diagnosis_date) ? ['nullable', 'date'] : ['required', 'date'];
        $familyHistoryRule = isset($latestAssessment->family_history_of_cancer) ? ['nullable', 'boolean'] : ['required', 'boolean'];
        $hypertensionRule = isset($latestAssessment->hypertension) ? ['nullable', 'boolean'] : ['required', 'boolean'];
        $asthmaRule = isset($latestAssessment->asthma) ? ['nullable', 'boolean'] : ['required', 'boolean'];
        $cirrhosisRule = isset($latestAssessment->cirrhosis) ? ['nullable', 'boolean'] : ['required', 'boolean'];
        $otherCancerHistoryRule = isset($latestAssessment->other_cancer_history) ? ['nullable', 'boolean'] : ['required', 'boolean'];
        // Add rules for new date fields (assuming they are not pre-filled generally)
        $beginningTreatmentDateRule = ['required', 'date', 'before_or_equal:today'];
        $endTreatmentDateRule = ['required', 'date', 'after_or_equal:beginning_of_treatment_date', 'before_or_equal:today'];


        $rules = [
            'age_at_diagnosis' => $ageRule,
            'gender' => $genderRule,
            'country' => ['required', 'string', 'max:255'],
            'diagnosis_date' => $diagnosisDateRule,
            'cancer_stage' => ['required', 'string', 'max:50'],
            'family_history_of_cancer' => $familyHistoryRule,
            'smoking_status' => ['required', 'string', 'max:100'],
            'bmi' => ['required', 'numeric', 'min:10', 'max:70', 'regex:/^\d+(\.\d{1,2})?$/'],
            'cholesterol_level' => ['required', 'integer', 'min:50', 'max:500'],
            'hypertension' => $hypertensionRule,
            'asthma' => $asthmaRule,
            'cirrhosis' => $cirrhosisRule,
            'other_cancer_history' => $otherCancerHistoryRule,
            'treatment_type' => ['required', 'string', 'max:255'],
            'beginning_of_treatment_date' => $beginningTreatmentDateRule,
            'end_treatment_date' => $endTreatmentDateRule,
            'notes' => ['nullable', 'string', 'max:5000'],
        ];

        $validatedData = $request->validate($rules);

        // --- Prepare data for FastAPI Payload ---
        $fastApiPayload = [];

        $fastApiPayload['age'] = (float)($validatedData['age_at_diagnosis'] ?? $latestAssessment->age_at_diagnosis);
        $fastApiPayload['gender'] = (int)($validatedData['gender'] ?? $latestAssessment->gender); // Already 0 or 1
        $fastApiPayload['diagnosis_date'] = Carbon::parse($validatedData['diagnosis_date'] ?? $latestAssessment->diagnosis_date)->format('Y-m-d');
        $fastApiPayload['beginning_of_treatment_date'] = Carbon::parse($validatedData['beginning_of_treatment_date'])->format('Y-m-d');
        $fastApiPayload['end_treatment_date'] = Carbon::parse($validatedData['end_treatment_date'])->format('Y-m-d');

        $cancerStageString = $validatedData['cancer_stage'];
        $stageMapping = [
            'Stage I' => 1, 'Stage IA' => 2, 'Stage IB' => 3, 'Stage II' => 4, 'Stage IIA' => 5,
            'Stage IIB' => 6, 'Stage III' => 7, 'Stage IIIA' => 8, 'Stage IIIB' => 9,
            'Stage IIIC' => 10, 'Stage IV' => 11, 'Stage IVA' => 12, 'Stage IVB' => 13,
            'Unknown' => 0 // Default for unknown, adjust if model expects different
        ];
        $fastApiPayload['cancer_stage'] = $stageMapping[$cancerStageString] ?? 0;

        $boolFieldsMap = [
            'family_history_of_cancer' => 'family_history',
            'hypertension' => 'hypertension',
            'asthma' => 'asthma',
            'cirrhosis' => 'cirrhosis',
            'other_cancer_history' => 'other_cancer'
        ];
        foreach ($boolFieldsMap as $formKey => $apiKey) {
            $value = $validatedData[$formKey] ?? ($latestAssessment ? $latestAssessment->$formKey : null);
            // Laravel's 'boolean' validation rule converts '1', 'true', 'on', 'yes' to true, and '0', 'false', 'off', 'no' to false.
            // So $value here will be true/false if validation passed.
            $fastApiPayload[$apiKey] = $value !== null ? (int)(bool)$value : 0;
        }

        $fastApiPayload['bmi'] = isset($validatedData['bmi']) ? (float)$validatedData['bmi'] : null;
        $fastApiPayload['cholesterol_level'] = isset($validatedData['cholesterol_level']) ? (int)$validatedData['cholesterol_level'] : null;

        $smokingStatus = $validatedData['smoking_status'];
        $fastApiPayload['smoke_Current_Smoker'] = (int)($smokingStatus === 'Current Smoker');
        $fastApiPayload['smoke_Former_Smoker'] = (int)($smokingStatus === 'Former Smoker');
        $fastApiPayload['smoke_Never_Smoked'] = (int)($smokingStatus === 'Never Smoked');
        $fastApiPayload['smoke_Passive_Smoker'] = (int)($smokingStatus === 'Passive Smoker');

        $treatmentType = $validatedData['treatment_type'];
        $fastApiPayload['treatment_Chemotherapy'] = (int)($treatmentType === 'Chemotherapy');
        $fastApiPayload['treatment_Combined'] = (int)($treatmentType === 'Chemoradiation');
        $fastApiPayload['treatment_Radiation'] = (int)($treatmentType === 'Radiation Therapy');
        $fastApiPayload['treatment_Surgery'] = (int)($treatmentType === 'Surgery');

        // --- FastAPI Call ---
        $fastApiServiceUrl = rtrim(config('services.fastapi.base_url', 'http://127.0.0.1:8000'), '/'); // Provide a default
        $predictionResult = null;
        $apiErrorDetails = null;
        $rawApiResponseForDb = null;


        try {
            Log::info('Calling Mortality Prediction API', ['url' => $fastApiServiceUrl . '/api/mortality-prediction/predict', 'payload' => $fastApiPayload]);
            $response = Http::timeout(60)->post($fastApiServiceUrl . '/api/mortality-prediction/predict', $fastApiPayload);
            $rawApiResponseForDb = ['status' => $response->status(), 'body' => $response->json() ?? $response->body()];


            if ($response->successful()) {
                $predictionResult = $response->json();
                if (!isset($predictionResult['survived'])) { // Check if critical key 'survived' is missing
                    $apiErrorDetails = ['error' => 'API response missing critical "survived" key.', 'response_received' => $predictionResult];
                    Log::error('Mortality Prediction API Success but malformed response', $apiErrorDetails);
                    $predictionResult = null; // Nullify to trigger error handling below
                } else {
                    Log::info('Mortality Prediction API Success', ['response' => $predictionResult]);
                }
            } else {
                $apiErrorDetails = $response->json() ?? ['error' => 'Unknown error structure from API', 'status' => $response->status(), 'raw_body' => $response->body()];
                Log::error('Mortality Prediction API Client/Server Error', [
                    'status' => $response->status(),
                    'body' => $apiErrorDetails,
                    'payload_sent' => $fastApiPayload
                ]);
            }
        } catch (\Illuminate\Http\Client\ConnectionException $e) {
            $apiErrorDetails = ['error' => 'Connection Exception', 'message' => $e->getMessage()];
            $rawApiResponseForDb = $apiErrorDetails;
            Log::error('Mortality Prediction API Connection Exception', ['exception_message' => $e->getMessage(), 'payload_sent' => $fastApiPayload]);
        } catch (\Exception $e) {
            $apiErrorDetails = ['error' => 'Generic Exception during API call', 'message' => $e->getMessage()];
            $rawApiResponseForDb = $apiErrorDetails;
            Log::error('Mortality Prediction API Generic Exception', ['exception_message' => $e->getMessage(), 'payload_sent' => $fastApiPayload]);
        }

        // --- Save Data and API Response to Database ---
        $dataToSave = $validatedData;
        $dataToSave['user_id'] = $user->id;

        // Ensure booleans for DB are correctly typed
        foreach ($boolFieldsMap as $formKey => $apiKey) {
             $value = $validatedData[$formKey] ?? ($latestAssessment ? $latestAssessment->$formKey : null);
             $dataToSave[$formKey] = $value !== null ? (bool)$value : null; // Save as boolean in DB
        }
         // Ensure other pre-filled fields are sourced correctly if not in $validatedData
        if (isset($latestAssessment->age_at_diagnosis) && !isset($validatedData['age_at_diagnosis'])) $dataToSave['age_at_diagnosis'] = $latestAssessment->age_at_diagnosis;
        if (isset($latestAssessment->gender) && !isset($validatedData['gender'])) $dataToSave['gender'] = $latestAssessment->gender; // Will be '0' or '1'
        if (isset($latestAssessment->diagnosis_date) && !isset($validatedData['diagnosis_date'])) $dataToSave['diagnosis_date'] = $latestAssessment->diagnosis_date;


        $dataToSave['api_response_payload'] = $rawApiResponseForDb; // Store raw response or error structure

        if ($predictionResult && isset($predictionResult['survived'])) {
            $dataToSave['predicted_survival_status'] = $predictionResult['survived'];
            $dataToSave['prediction_value'] = $predictionResult['prediction']; // int 0 or 1
            $dataToSave['prediction_probability_class_0'] = $predictionResult['prediction_probability_class_0']; // float
            $dataToSave['prediction_probability_class_1'] = $predictionResult['prediction_probability_class_1']; // float
            $dataToSave['api_error_message'] = null;
        } else {
            // Ensure these fields are at least nullable in DB or set to default if error
            $dataToSave['predicted_survival_status'] = null;
            $dataToSave['prediction_value'] = null;
            $dataToSave['prediction_probability_class_0'] = null;
            $dataToSave['prediction_probability_class_1'] = null;
            $dataToSave['api_error_message'] = is_string($apiErrorDetails) ? $apiErrorDetails : json_encode($apiErrorDetails);
        }


        try {
            $newRecord = PatientSurvivalPrediction::create($dataToSave);
            $sessionDataForView = $predictionResult ?? $apiErrorDetails ?? [];

            if ($predictionResult && isset($predictionResult['survived'])) {
                $successMsg = 'Patient data saved. Prediction: ' . $predictionResult['survived'] .
                              ' (Prob Won\'t Survive: ' . round($predictionResult['prediction_probability_class_0'] * 100, 1) . '%, '.
                              'Prob Will Survive: ' . round($predictionResult['prediction_probability_class_1'] * 100, 1) . '%)';
                return redirect()->route('psp.create')->with('success', $successMsg)
                                 ->with('predictionDetails', $sessionDataForView);
            } elseif ($apiErrorDetails) {
                 $errorMessage = 'Patient data saved, but prediction failed. ';
                 if(is_array($apiErrorDetails) && isset($apiErrorDetails['error'])) {
                    $errorMessage .= $apiErrorDetails['error'] . (isset($apiErrorDetails['message']) ? ': '.$apiErrorDetails['message'] : '');
                    if(isset($apiErrorDetails['detail'])) $errorMessage .= ' Details: '. (is_string($apiErrorDetails['detail']) ? $apiErrorDetails['detail'] : json_encode($apiErrorDetails['detail']));
                 } elseif(is_string($apiErrorDetails)) {
                    $errorMessage .= $apiErrorDetails;
                 } else {
                    $errorMessage .= 'Please check system logs.';
                 }
                 return back()->withInput()->with('error', $errorMessage)->with('predictionDetails', $sessionDataForView);
            } else {
                 return redirect()->route('psp.create')->with('warning', 'Patient data saved, but prediction result was inconclusive or not returned by the service.')
                                 ->with('predictionDetails', $sessionDataForView);
            }

        } catch (\Exception $e) {
            Log::critical('Failed to save patient survival data to DB: ' . $e->getMessage(), ['exception_trace' => $e->getTraceAsString(), 'data_attempted' => $dataToSave]);
            $dbError = 'Critical error: Failed to save patient data due to a database issue. Please contact support. ';
            if ($apiErrorDetails) {
                $dbError .= 'The prediction service also reported an issue. ';
            }
            return back()->withInput()->with('error', $dbError);
        }
    }

    private function getCountryOptions(): array
    {
        // Basic list, consider a more comprehensive solution for production
        return [
            'Tunisia' => 'Tunisia',
            'United States' => 'United States',
            'Canada' => 'Canada',
            'United Kingdom' => 'United Kingdom',
            'Germany' => 'Germany',
            'France' => 'France',
            'Other' => 'Other',
        ];
    }
}