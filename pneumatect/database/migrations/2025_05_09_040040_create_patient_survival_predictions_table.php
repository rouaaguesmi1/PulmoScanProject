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
        Schema::create('patient_survival_predictions', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->onDelete('cascade'); // Link to the user (patient)

            // Patient Demographics & History (could be pre-filled or entered)
            $table->integer('age_at_diagnosis');
            $table->string('gender'); // e.g., Male, Female, Other
            $table->string('country')->nullable(); // Consider a separate countries table or standard list
            $table->date('diagnosis_date');
            $table->string('cancer_stage'); // e.g., Stage I, Stage IIA, etc.
            $table->boolean('family_history_of_cancer'); // true for yes, false for no

            // Lifestyle & Comorbidities
            $table->string('smoking_status'); // e.g., Current Smoker, Former Smoker, Never Smoked, Passive Smoker
            $table->decimal('bmi', 5, 2)->nullable(); // Body Mass Index, e.g., 24.50
            $table->integer('cholesterol_level')->nullable(); // e.g., mg/dL
            $table->boolean('hypertension');
            $table->boolean('asthma');
            $table->boolean('cirrhosis');
            $table->boolean('other_cancer_history'); // History of other cancers

            // Treatment
            $table->string('treatment_type'); // e.g., Surgery, Chemotherapy, Radiation, Combined, Palliative, None

            // Prediction Results (to be filled by your model later)
            $table->string('predicted_survival_status')->nullable(); // e.g., Alive, Deceased
            $table->integer('predicted_survival_months')->nullable();
            $table->float('prediction_confidence', 8, 4)->nullable();
            $table->json('prediction_details')->nullable(); // For full API response or extra data

            $table->text('notes')->nullable(); // Optional notes for this record
            $table->timestamps(); // created_at, updated_at
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('patient_survival_predictions');
    }
};
