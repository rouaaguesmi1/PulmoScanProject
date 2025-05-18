<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class LungCancerRiskAssessment extends Model
{
    use HasFactory;

    protected $fillable = [
        'user_id', 'age', 'gender', 'air_pollution', 'alcohol_use', 
        'dust_allergy', 'occupational_hazards', 'genetic_risk', 
        'chronic_lung_disease', 'balanced_diet', 'obesity', 'smoking', 
        'passive_smoker', 'chest_pain', 'coughing_of_blood', 'fatigue', 
        'weight_loss', 'shortness_of_breath', 'wheezing', 
        'swallowing_difficulty', 'clubbing_of_finger_nails',
        'predicted_risk_level', 'prediction_score', 'notes', 'assessment_date',
    ];

    // If you store categorical values as numbers but want to map them to text,
    // you can use accessors or enums (PHP 8.1+).
    // Example for gender if stored as numbers (0 for Male, 1 for Female)
    // public function getGenderAttribute($value) {
    //     return $value == 0 ? 'Male' : ($value == 1 ? 'Female' : 'Other');
    // }

    protected $casts = [
        'assessment_date' => 'datetime',
        'prediction_score' => 'float',
        // Cast integer fields if they come as strings from form
        'age' => 'integer', 
        'air_pollution' => 'integer',
        // ... cast other tinyInteger fields as 'integer' if needed ...
    ];

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }
}