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
            // Determine a suitable column to add these after, e.g., after 'notes'
            // Or add them at the end if no specific order is critical
            $table->integer('prediction_value')->nullable()->after('predicted_survival_status');
            $table->float('prediction_probability_class_0', 10, 7)->nullable()->after('prediction_value'); // Precision 10, 7 decimal places
            $table->float('prediction_probability_class_1', 10, 7)->nullable()->after('prediction_probability_class_0');
            $table->json('api_response_payload')->nullable()->after('prediction_probability_class_1'); // For storing the full API JSON response
            $table->text('api_error_message')->nullable()->after('api_response_payload');    // For storing API error messages
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('patient_survival_predictions', function (Blueprint $table) {
            $table->dropColumn([
                'predicted_survival_status',
                'prediction_value',
                'prediction_probability_class_0',
                'prediction_probability_class_1',
                'api_response_payload',
                'api_error_message'
            ]);
        });
    }
};