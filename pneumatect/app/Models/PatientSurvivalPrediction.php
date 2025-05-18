<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Carbon\Carbon;

class PatientSurvivalPrediction extends Model
{
    use HasFactory;

    protected $table = 'patient_survival_predictions'; // Explicitly define table name

    protected $fillable = [
        'user_id',
        'age_at_diagnosis',
        'gender', // Stored as '0' or '1' (or your int mapping)
        'country',
        'diagnosis_date',
        'cancer_stage',
        'family_history_of_cancer', // boolean
        'smoking_status',
        'bmi',
        'cholesterol_level',
        'hypertension', // boolean
        'asthma', // boolean
        'cirrhosis', // boolean
        'other_cancer_history', // boolean
        'treatment_type',
        'beginning_of_treatment_date',
        'end_treatment_date',
        'notes',
        // New fields for API response
        'predicted_survival_status', // string e.g. "Will Survive"
        'prediction_value',          // int e.g. 0 or 1
        'prediction_probability_class_0', // float
        'prediction_probability_class_1', // float
        'api_response_payload',       // json/array
        'api_error_message',          // text/string
    ];

    protected $casts = [
        'diagnosis_date' => 'date:Y-m-d',
        'beginning_of_treatment_date' => 'date:Y-m-d',
        'end_treatment_date' => 'date:Y-m-d',
        'family_history_of_cancer' => 'boolean',
        'hypertension' => 'boolean',
        'asthma' => 'boolean',
        'cirrhosis' => 'boolean',
        'other_cancer_history' => 'boolean',
        'bmi' => 'float',
        'cholesterol_level' => 'integer',
        'age_at_diagnosis' => 'integer',
        // 'gender' will be stored as string '0' or '1' from form, or your integer mapping. Cast to int if needed upon retrieval.
        // Or, ensure it's always int before saving if $genderOptions keys are int.
        'gender' => 'integer',


        'api_response_payload' => 'array',
        'prediction_probability_class_0' => 'float',
        'prediction_probability_class_1' => 'float',
        'prediction_value' => 'integer',
    ];

    /**
     * Get the user that owns the prediction.
     */
    public function user()
    {
        return $this->belongsTo(User::class);
    }

    // Optional: Accessor to ensure Carbon instance for dates when retrieved,
    // though the 'date:Y-m-d' cast often handles this for standard operations.
    public function getDiagnosisDateAttribute($value)
    {
        return $value ? Carbon::parse($value) : null;
    }

    public function getBeginningOfTreatmentDateAttribute($value)
    {
        return $value ? Carbon::parse($value) : null;
    }

    public function getEndOfTreatmentDateAttribute($value)
    {
        return $value ? Carbon::parse($value) : null;
    }
}