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
        Schema::create('subtype_predictions', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->onDelete('cascade'); // Link to the user who uploaded
            $table->string('image_path'); // Path where the image is stored
            $table->string('original_filename')->nullable(); // Optional: Store original name
            $table->string('predicted_class')->nullable(); // Result from FastAPI
            $table->float('confidence', 8, 4)->nullable(); // Result from FastAPI (adjust precision if needed)
            $table->json('prediction_details')->nullable(); // Optional: Store full API response or other info
            $table->timestamps(); // created_at and updated_at
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('subtype_predictions');
    }
};
