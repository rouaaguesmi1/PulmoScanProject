<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::table('patient_survival_predictions', function (Blueprint $table) {
            // Add after an existing column, e.g., after 'treatment_type' or 'notes'
            // Choose an appropriate column to add them after based on your table structure.
            $table->date('beginning_of_treatment_date')->nullable()->after('treatment_type');
            $table->date('end_treatment_date')->nullable()->after('beginning_of_treatment_date');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('patient_survival_predictions', function (Blueprint $table) {
            $table->dropColumn(['beginning_of_treatment_date', 'end_treatment_date']);
        });
    }
};