<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Carbon\Carbon;

class SmokingCessationProfile extends Model
{
    use HasFactory;

    protected $fillable = [
        'user_id',
        'quit_date',
        'cigarettes_per_day_before',
        'cost_per_pack',
        'pack_size',
    ];

    protected $casts = [
        'quit_date' => 'date',
        'cost_per_pack' => 'decimal:2',
        'cigarettes_per_day_before' => 'integer',
        'pack_size' => 'integer',
    ];

    /**
     * Get the user that owns this profile.
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    /**
     * Calculate days smoke-free.
     */
    public function getDaysSmokeFreeAttribute(): int
    {
        return Carbon::parse($this->quit_date)->diffInDays(Carbon::now());
    }

    /**
     * Calculate cigarettes avoided.
     */
    public function getCigarettesAvoidedAttribute(): int
    {
        return $this->days_smoke_free * $this->cigarettes_per_day_before;
    }

    /**
     * Calculate cost per cigarette.
     */
    public function getCostPerCigaretteAttribute(): float
    {
        if ($this->pack_size > 0) {
            return $this->cost_per_pack / $this->pack_size;
        }
        return 0;
    }

    /**
     * Calculate money saved.
     */
    public function getMoneySavedAttribute(): float
    {
        return $this->cigarettes_avoided * $this->cost_per_cigarette;
    }
}