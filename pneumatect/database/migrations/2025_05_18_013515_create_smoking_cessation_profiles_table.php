<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('smoking_cessation_profiles', function (Blueprint $table) {
        $table->id();
        $table->foreignId('user_id')->constrained()->onDelete('cascade')->unique(); // Each user has one profile
        $table->date('quit_date');
        $table->unsignedInteger('cigarettes_per_day_before');
        $table->decimal('cost_per_pack', 8, 2);
        $table->unsignedInteger('pack_size')->default(20); // e.g., 20 cigarettes in a pack
        $table->timestamps();
    });
    }

    public function down(): void
    {
        Schema::dropIfExists('smoking_cessation_profiles');
    }
};