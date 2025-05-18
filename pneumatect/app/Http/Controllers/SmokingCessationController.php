<?php

namespace App\Http\Controllers;

use App\Models\SmokingCessationProfile;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Carbon\Carbon;

class SmokingCessationController extends Controller
{
    /**
     * Display the smoking cessation tracker page.
     * Shows setup form or tracker dashboard.
     */
    public function index()
    {
        $user = Auth::user();
        $profile = $user->smokingCessationProfile()->first(); // Use relation if defined in User model, or query directly

        // If User model doesn't have the hasOne relation yet, define it or query like:
        // $profile = SmokingCessationProfile::where('user_id', $user->id)->first();


        if ($profile) {
            // Data is already calculated by accessors in the model
            return view('smoking_cessation.index', compact('profile'));
        }

        // No profile, show setup part of the view
        return view('smoking_cessation.index', ['profile' => null]);
    }

    /**
     * Store or update the smoking cessation profile.
     */
    public function store(Request $request)
    {
        $user = Auth::user();

        $validatedData = $request->validate([
            'quit_date' => 'required|date|before_or_equal:today',
            'cigarettes_per_day_before' => 'required|integer|min:1|max:200', // Max 200 as a sensible limit
            'cost_per_pack' => 'required|numeric|min:0.01|max:1000', // Max 1000 as a sensible limit
            'pack_size' => 'required|integer|min:1|max:100', // Max 100 as a sensible limit
        ]);

        try {
            SmokingCessationProfile::updateOrCreate(
                ['user_id' => $user->id],
                $validatedData
            );

            return redirect()->route('smoking_cessation.index')->with('success', 'Your smoking cessation profile has been saved!');
        } catch (\Exception $e) {
            \Illuminate\Support\Facades\Log::error('Smoking Cessation Profile save error: ' . $e->getMessage());
            return redirect()->route('smoking_cessation.index')
                       ->with('error', 'There was an error saving your profile: ' . $e->getMessage())
                       ->withInput();
        }
    }
}