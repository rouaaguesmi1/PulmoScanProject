<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\ContactController;
use App\Http\Controllers\PredictionController;
use App\Http\Controllers\SupportController;
use App\Http\Controllers\LungCancerRiskController;
use App\Http\Controllers\PatientSurvivalController;
use App\Http\Controllers\DashboardController;
use App\Http\Controllers\ImageEnhancementController;
use App\Http\Controllers\NoduleDetectionController;
use App\Http\Controllers\NoduleClassificationController;
use App\Http\Controllers\PatientCancerClassificationController;
use App\Http\Controllers\AppointmentController;
use App\Http\Controllers\SmokingCessationController;
use App\Http\Controllers\DicomViewerController;


Route::get('/', function () {
    return view('welcome');
});

Route::middleware([
    'auth:sanctum',
    config('jetstream.auth_session'),
    'verified',
])->group(function () {
    Route::get('/dashboard', function () {
        return view('dashboard');
    })->name('dashboard');

    Route::get('/dashboard', [DashboardController::class, 'index'])->name('dashboard'); // Updated
    Route::get('/dashboard/subtype-prediction', [PredictionController::class, 'subtypePredictionCreate'])
         ->name('predict.subtype.create'); 

    // Route to handle the form submission
Route::post('/dashboard/subtype-prediction', [PredictionController::class, 'storeSubtypePrediction'])
    ->name('predict.subtype.store'); 


Route::get('/dashboard/medical-history', [PredictionController::class, 'medicalHistoryIndex'])
    ->name('medical-history.index');


Route::get('/medical-history/pdf', [PredictionController::class, 'generatePdf'])
->name('medical_history.pdf');

Route::get('/dashboard/help-center', function () {
    return view('help_center/index');
})->name('help_center.index');


Route::get('/dashboard/contact-support', [SupportController::class, 'create'])->name('contact-support.create');
Route::post('/dashboard/contact-support', [SupportController::class, 'store'])->name('contact-support.store'); // For form submission


Route::get('/dashboard/lung-cancer-risk/create', [LungCancerRiskController::class, 'create'])->name('lcr.create');
Route::post('/dashboard/lung-cancer-risk', [LungCancerRiskController::class, 'store'])->name('lcr.store');


Route::get('/dashboard/patient-survival-prediction/create', [PatientSurvivalController::class, 'create'])->name('psp.create');
Route::post('/dashboard/patient-survival-prediction', [PatientSurvivalController::class, 'store'])->name('psp.store');


Route::get('/dashboard/wellness/quit-smoking-guide', function () {
    return view('wellness/quit_smoking_guide');
})->name('wellness.quit-smoking-guide');

Route::get('/dashboard/wellness/diet', function () {
    return view('wellness/diet_nutrition_guide');
})->name('wellness.diet-nutrition-guide');

Route::get('/dashboard/wellness/exercise', function () {
    return view('wellness/recommended_exercises_guide');
})->name('wellness.exercise-guide');

Route::get('/dashboard/wellness/mental', function () {
    return view('wellness/mental_well_being_guide');
})->name('wellness.mental-guide');


Route::get('/dashboard/wellness/lungcancer', function () {
    return view('wellness/understanding_lung_cancer_guide');
})->name('wellness.lungcancer-guide');

Route::get('/dashboard/image-enhancement', [ImageEnhancementController::class, 'showEnhancementForm'])
    ->name('image.enhancement.form');

Route::post('/dashboard/image-enhancement', [ImageEnhancementController::class, 'processImageEnhancement'])
    ->name('image.enhancement.process');


Route::get('/dashboard/nodule-detection', [NoduleDetectionController::class, 'showDetectionForm'])
    ->name('nodule.detection.form');

Route::post('/dashboard/nodule-detection', [NoduleDetectionController::class, 'processNoduleDetection'])
    ->name('nodule.detection.process');

Route::get('/dashboard/nodule-classification', [NoduleClassificationController::class, 'showClassificationForm'])
    ->name('nodule.classification.form');

Route::post('/dashboard/nodule-classification', [NoduleClassificationController::class, 'processNoduleClassification'])
    ->name('nodule.classification.process');

Route::get('/dashboard/patient-cancer-classification', [PatientCancerClassificationController::class, 'showPredictionForm'])
         ->name('patient.cancer.classification.form');

    Route::post('/dashboard/patient-cancer-classification', [PatientCancerClassificationController::class, 'processPatientPrediction'])
         ->name('patient.cancer.classification.process');

Route::get('/dashboard/appointments/create', [AppointmentController::class, 'create'])->name('appointments.create');
    Route::post('/dashboard/appointments', [AppointmentController::class, 'store'])->name('appointments.store');    


Route::get('/dashboard/my-appointments', [AppointmentController::class, 'index'])->name('appointments.index');

   Route::get('/smoking-cessation', [SmokingCessationController::class, 'index'])->name('smoking_cessation.index');
    Route::post('/smoking-cessation', [SmokingCessationController::class, 'store'])->name('smoking_cessation.store');

Route::get('/dicom-viewer', [DicomViewerController::class, 'showViewer'])->name('dicom.viewer');
});

Route::get('/terms', function () {
    return view('terms');
})->name('terms');
Route::get('/privacy', function () {
    return view('policy');
})->name('privacy');

Route::get('/services', function () {
    return view('services');
})->name('services');


Route::get('/pricing', function () {
    return view('pricing');
})->name('pricing');

Route::post('/contact/send', [ContactController::class, 'send'])->name('contact.send');
Route::get('/contact', function () {
    return view('contact'); // or however you're rendering the contact form page
})->name('contact');

Route::get('/about', function () {
    return view('about');
})->name('about');
Route::get('/services', function () {
    return view('services');
})->name('services');
Route::get('/blog', function () {
    return view('blog');
})->name('blog');


Route::get('nodule-classification', function () {
    return view('services/service-nodule-classification');
})->name('services.nodule-classification');

Route::get('nodule-detection', function () {
    return view('services/service-nodule-detection');
})->name('services.nodule-detection');

Route::get('subtype-classification', function () {
    return view('services/service-subtype-classification');
})->name('services.subtype-classification');


Route::get('stage-estimation', function () {
    return view('services/service-stage-estimation');
})->name('services.stage-estimation');

Route::get('risk-prediction', function () {
    return view('services/service-risk-prediction');
})->name('services.risk-prediction');

Route::get('ct-enhancement', function () {
    return view('services/service-ct-enhancement');
})->name('services.ct-enhancement');

Route::get('lung-function', function () {
    return view('services/service-lung-function');
})->name('services.lung-function');

Route::get('mortality-prediction', function () {
    return view('services/service-mortality-prediction');
})->name('services.mortality-prediction');


Route::get('blogOne', function () {
    return view('blog/blogOne');
})->name('blogOne');
Route::get('blogTwo', function () {
    return view('blog/blogTwo');
})->name('blogTwo');
Route::get('blogThree', function () {
    return view('blog/blogThree');
})->name('blogThree');

Route::get('OurApi', function () {
    return view('OurApi');
})->name('OurApi');


