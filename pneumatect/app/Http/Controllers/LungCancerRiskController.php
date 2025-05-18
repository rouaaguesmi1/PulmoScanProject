<?php

namespace App\Http\Controllers;

use App\Models\LungCancerRiskAssessment; // Assuming you have this model
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Log;
use Illuminate\View\View;

class LungCancerRiskController extends Controller
{
    public function create(): View
    {
        $user = Auth::user();
        // Fetch the latest assessment for this user to get previous age/gender
        $latestAssessment = LungCancerRiskAssessment::where('user_id', $user->id)
                                                    ->orderBy('created_at', 'desc')
                                                    ->first();

        // Define options for select dropdowns (example for levels 1-9)
        // You MUST adjust these scales and labels based on your model's actual expected input.
        $levelOptions = [];
        for ($i = 1; $i <= 9; $i++) { 
            $levelOptions[$i] = "Level " . $i; // Example: Level 1 (Low) to Level 9 (High)
        }

        $genderOptions = [
            'Male' => 'Male',
            'Female' => 'Female',
            // 'Other' => 'Other', // Add if your model/database supports it
        ];

        return view('risk_assessment.lung_cancer_create', [
            'levelOptions' => $levelOptions,
            'genderOptions' => $genderOptions,
            'latestAge' => $latestAssessment->age ?? null,       // Pass user's last recorded age
            'latestGender' => $latestAssessment->gender ?? null, // Pass user's last recorded gender
        ]);
    }

    public function store(Request $request): RedirectResponse
    {
        $user = Auth::user();
        $latestAssessment = LungCancerRiskAssessment::where('user_id', $user->id)
                                                    ->orderBy('created_at', 'desc')
                                                    ->first();

        // If age/gender were pre-filled (readonly/disabled), they might not be in $request
        // or we want to ensure the pre-filled value is used.
        $ageFromRequest = $request->input('age');
        $genderFromRequest = $request->input('gender'); // This will come from hidden input if select was disabled

        // Determine if age/gender should be validated as 'required' or 'nullable'
        $ageRule = isset($latestAssessment->age) ? ['nullable', 'integer', 'min:1', 'max:120'] : ['required', 'integer', 'min:1', 'max:120'];
        $genderRule = isset($latestAssessment->gender) ? ['nullable', 'string', 'in:Male,Female'] : ['required', 'string', 'in:Male,Female'];


        $rules = [
            'age' => $ageRule,
            'gender' => $genderRule,
            'air_pollution' => ['required', 'integer', 'min:1', 'max:8'],
            'alcohol_use' => ['required', 'integer', 'min:1', 'max:8'],
            'dust_allergy' => ['required', 'integer', 'min:1', 'max:8'],
            'occupational_hazards' => ['required', 'integer', 'min:1', 'max:8'],
            'genetic_risk' => ['required', 'integer', 'min:1', 'max:7'],
            'chronic_lung_disease' => ['required', 'integer', 'min:1', 'max:7'],
            'balanced_diet' => ['required', 'integer', 'min:1', 'max:7'],
            'obesity' => ['required', 'integer', 'min:1', 'max:7'],
            'smoking' => ['required', 'integer', 'min:1', 'max:8'],
            'passive_smoker' => ['required', 'integer', 'min:1', 'max:8'],
            'chest_pain' => ['required', 'integer', 'min:1', 'max:9'],
            'coughing_of_blood' => ['required', 'integer', 'min:1', 'max:9'],
            'fatigue' => ['required', 'integer', 'min:1', 'max:9'],
            'weight_loss' => ['required', 'integer', 'min:1', 'max:8'],
            'shortness_of_breath' => ['required', 'integer', 'min:1', 'max:9'],
            'wheezing' => ['required', 'integer', 'min:1', 'max:9'],
            'swallowing_difficulty' => ['required', 'integer', 'min:1', 'max:8'],
            'clubbing_of_finger_nails' => ['required', 'integer', 'min:1', 'max:9'],
            'notes' => ['nullable', 'string', 'max:2000'],
        ];

        $validatedData = $request->validate($rules);
        
        $dataToSave = $validatedData;
        $dataToSave['user_id'] = $user->id;
        $dataToSave['assessment_date'] = now();

        // If age was pre-filled (readonly), it's in $validatedData.
        // If it wasn't pre-filled, it was required, so it's in $validatedData.
        // So $validatedData['age'] should be correct.

        // If gender was pre-filled (disabled select, submitted via hidden input),
        // it's in $validatedData.
        // If it wasn't pre-filled (select was enabled), it was required, so it's in $validatedData.
        // So $validatedData['gender'] should be correct.

        try {
            LungCancerRiskAssessment::create($dataToSave);
            return redirect()->route('lcr.create')->with('success', 'Risk assessment data saved successfully.');
        } catch (\Exception $e) {
            Log::error('Failed to save lung cancer risk assessment: ' . $e->getMessage(), ['exception' => $e, 'data' => $dataToSave]);
            return back()->withInput()->with('error', 'Failed to save assessment data. Please try again.');
        }
    }
}