<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class SubtypePrediction extends Model
{
    use HasFactory;

    protected $fillable = [
        'user_id',
        'image_path',
        'original_filename',
        'predicted_class',
        'confidence',
        'prediction_details', // Add if using the JSON column
    ];

    // Optional: Cast JSON column if you added it
     protected $casts = [
         'prediction_details' => 'array',
         'confidence' => 'float', // Ensure confidence is treated as float
         'last_login_at' => 'datetime', // Example from previous request
     ];

    /**
     * Get the user that owns the prediction.
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }
}