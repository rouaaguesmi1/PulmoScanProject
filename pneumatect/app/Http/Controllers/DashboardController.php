<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\View\View;
use App\Models\SubtypePrediction;
use App\Models\LungCancerRiskAssessment;
use App\Models\PatientSurvivalPrediction;
use Carbon\Carbon;

class DashboardController extends Controller
{
    public function index(): View
    {
        $user = Auth::user();

        // Fetch latest records for the user
        $latestSubtypePrediction = SubtypePrediction::where('user_id', $user->id)
            ->orderBy('created_at', 'desc')
            ->first();

        $latestRiskAssessment = LungCancerRiskAssessment::where('user_id', $user->id)
            ->orderBy('assessment_date', 'desc') // Assuming assessment_date or created_at
            ->first();

        $latestSurvivalPredictionInput = PatientSurvivalPrediction::where('user_id', $user->id)
            ->orderBy('created_at', 'desc')
            ->first(); // This is the input data, not the actual survival prediction yet

        // Prepare data points for the dashboard
        // CORRECTED LINE 33: Added nullsafe operator ?->
        $bmi = $latestSurvivalPredictionInput?->bmi ?? ($latestRiskAssessment?->obesity ? 'High (based on risk factor)' : null);

        // CORRECTED: Added nullsafe operator ?->
        $shortnessOfBreathLevel = $latestRiskAssessment?->shortness_of_breath ?? null;
        // CORRECTED: Added nullsafe operator ?->
        $smokingStatus = $latestRiskAssessment?->smoking ?? ($latestSurvivalPredictionInput?->smoking_status ?? null);


        // Example: Prepare data for a simple "Recent Activity" list
        $recentActivities = collect();
        if ($latestSubtypePrediction) {
            $recentActivities->push([
                'type' => 'CT Scan Prediction',
                'date' => $latestSubtypePrediction->created_at,
                'detail' => "Result: {$latestSubtypePrediction->predicted_class} (Confidence: " . number_format(($latestSubtypePrediction->confidence ?? 0) * 100, 1) . "%)",
                'link' => route('medical-history.index') // Link to history page
            ]);
        }
        if ($latestRiskAssessment) {
            // Note: $smokingStatus and $shortnessOfBreathLevel used here are already safely derived above
            $recentActivities->push([
                'type' => 'Lung Cancer Risk Assessment',
                'date' => $latestRiskAssessment->assessment_date ?? $latestRiskAssessment->created_at,
                'detail' => "Smoking: " . ($smokingStatus ?? 'N/A') . ", SOB: " . ($shortnessOfBreathLevel ?? 'N/A'), // Ensure display is N/A if null
                'link' => route('medical-history.index')
            ]);
        }
        if ($latestSurvivalPredictionInput) {
            $recentActivities->push([
                'type' => 'Patient Data Submitted (Survival)',
                'date' => $latestSurvivalPredictionInput->created_at,
                'detail' => "Stage: " . ($latestSurvivalPredictionInput->cancer_stage ?? 'N/A') . ", Treatment: " . ($latestSurvivalPredictionInput->treatment_type ?? 'N/A'),
                'link' => route('medical-history.index')
            ]);
        }
        $recentActivities = $recentActivities->sortByDesc('date')->take(5); // Get latest 5 activities


        // Placeholder for chart data - In a real app, this would be more complex
        $assessmentTrendData = [
            'labels' => ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], // Example labels
            'sob_data' => [2, 3, 3, 4, 3, 5], // Example: Shortness of Breath levels
            'fatigue_data' => [4, 3, 5, 4, 5, 4], // Example: Fatigue levels
        ];


        return view('dashboard', [
            'user' => $user,
            'latestSubtypePrediction' => $latestSubtypePrediction,
            'latestRiskAssessment' => $latestRiskAssessment,
            'latestSurvivalPredictionInput' => $latestSurvivalPredictionInput,
            'bmi' => $bmi,
            'shortnessOfBreathLevel' => $shortnessOfBreathLevel,
            'smokingStatus' => $smokingStatus,
            'recentActivities' => $recentActivities,
            'assessmentTrendData' => $assessmentTrendData,
            // Add more data as needed for other stats/charts
        ]);
    }
}