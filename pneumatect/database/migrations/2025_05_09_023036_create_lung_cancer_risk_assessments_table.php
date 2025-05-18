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
        Schema::create('lung_cancer_risk_assessments', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->onDelete('cascade');
            $table->integer('age');
            $table->string('gender'); // Or tinyInteger if mapping to numeric IDs
            // For categorical data, assuming an integer scale (e.g., 1-7 or similar)
            // You'll need to define what these levels mean in your UI
            $table->tinyInteger('air_pollution')->nullable();
            $table->tinyInteger('alcohol_use')->nullable();
            $table->tinyInteger('dust_allergy')->nullable();
            $table->tinyInteger('occupational_hazards')->nullable();
            $table->tinyInteger('genetic_risk')->nullable();
            $table->tinyInteger('chronic_lung_disease')->nullable();
            $table->tinyInteger('balanced_diet')->nullable();
            $table->tinyInteger('obesity')->nullable();
            $table->tinyInteger('smoking')->nullable();
            $table->tinyInteger('passive_smoker')->nullable();
            $table->tinyInteger('chest_pain')->nullable();
            $table->tinyInteger('coughing_of_blood')->nullable();
            $table->tinyInteger('fatigue')->nullable();
            $table->tinyInteger('weight_loss')->nullable();
            $table->tinyInteger('shortness_of_breath')->nullable();
            $table->tinyInteger('wheezing')->nullable();
            $table->tinyInteger('swallowing_difficulty')->nullable();
            $table->tinyInteger('clubbing_of_finger_nails')->nullable();

            // Fields for storing the prediction result (add later or make nullable now)
            $table->string('predicted_risk_level')->nullable(); // e.g., 'Low', 'Medium', 'High'
            $table->float('prediction_score', 8, 4)->nullable(); // e.g., 0.0 to 1.0

            $table->text('notes')->nullable(); // Optional field for any notes
            $table->timestamp('assessment_date')->useCurrent(); // Or just use created_at
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('lung_cancer_risk_assessments');
    }
};
