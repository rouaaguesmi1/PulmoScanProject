<?php

namespace App\Http\Controllers;

use App\Models\Appointment;
use Illuminate\Http\Request;
  use Illuminate\Support\Facades\Auth;
use Carbon\Carbon;
use Illuminate\Support\Facades\Validator;
use Illuminate\Support\Str; // Import Str
class AppointmentController extends Controller
{


    public function index()
    {
        $userAppointments = Auth::user()->appointments()->orderBy('appointment_datetime', 'asc')->get();
        $calendarEvents = [];

        foreach ($userAppointments as $appointment) {
            $title = 'Appt: ' . $appointment->practitioner_name;
            // Optionally add part of the reason to the title if it exists
            // if ($appointment->reason) {
            //     $title .= ' (' . Str::limit($appointment->reason, 20) . ')';
            // }

            // Determine event color based on status (optional, uses Falcon's soft colors)
            $className = 'bg-soft-primary'; // Default for scheduled
            if ($appointment->status === 'completed') {
                $className = 'bg-soft-success';
            } elseif (Str::contains($appointment->status, 'cancelled')) {
                $className = 'bg-soft-danger text-decoration-line-through';
            }


            $calendarEvents[] = [
                'id' => $appointment->id,
                'title' => $title,
                'start' => $appointment->appointment_datetime->toIso8601String(),
                // 'end' => $appointment->appointment_datetime->addHours(1)->toIso8601String(), // Example: if appointments are 1 hour long
                'className' => $className,
                'extendedProps' => [
                    'practitioner_type' => $appointment->practitioner_type,
                    'practitioner_name' => $appointment->practitioner_name,
                    'full_reason' => $appointment->reason ?: 'No reason provided.',
                    'status' => Str::title(str_replace('_', ' ', $appointment->status)),
                    'formatted_date' => $appointment->appointment_datetime->format('D, M j, Y'),
                    'formatted_time' => $appointment->appointment_datetime->format('h:i A'),
                    'raw_datetime' => $appointment->appointment_datetime->toDateTimeString(),
                ]
            ];
        }

        return view('appointments.calendar', [
            'calendarEvents' => $calendarEvents,
        ]);
    }
    // Fake practitioner data
    private function getPractitionerData()
    {
        return [
            'types' => ['Lung Doctor', 'Therapist'],
            'by_type' => [
                'Lung Doctor' => [
                    'Dr. Eva Müller (Pulmonologist)',
                    'Dr. John Smith (Lung Specialist)',
                    'Dr. Aisha Khan (Respiratory Physician)',
                    'Dr. Kenji Tanaka (Thoracic Health Expert)'
                ],
                'Therapist' => [
                    'Ms. Sarah Jones (Counselor & Mental Wellness Coach)',
                    'Mr. David Lee (Psychotherapist & Stress Management)',
                    'Mrs. Emily White (Clinical Therapist for Chronic Illness Support)',
                    'Mr. Omar Hassan (Behavioral Health Specialist)'
                ]
            ]
        ];
    }

    /**
     * Show the form for creating a new appointment.
     *
     * @return \Illuminate\View\View
     */
    public function create()
    {
        $practitionerData = $this->getPractitionerData();
        return view('appointments.create', [
            'practitionerTypes' => $practitionerData['types'],
            'practitionersByType' => $practitionerData['by_type'],
        ]);
    }

    /**
     * Store a newly created appointment in storage.
     *
     * @param  \Illuminate\Http\Request  $request
     * @return \Illuminate\Http\RedirectResponse
     */
    public function store(Request $request)
    {
        $practitionerData = $this->getPractitionerData();

        $validator = Validator::make($request->all(), [
            'practitioner_type' => ['required', 'string', \Illuminate\Validation\Rule::in($practitionerData['types'])],
            'practitioner_name' => ['required', 'string', function ($attribute, $value, $fail) use ($request, $practitionerData) {
                $type = $request->input('practitioner_type');
                if (!isset($practitionerData['by_type'][$type]) || !in_array($value, $practitionerData['by_type'][$type])) {
                    $fail('The selected ' . strtolower(str_replace('_', ' ', $attribute)) . ' is not valid for the chosen type.');
                }
            }],
            'appointment_date' => 'required|date|after_or_equal:today',
            'appointment_time' => 'required|date_format:H:i',
            'reason' => 'nullable|string|max:1000',
        ]);

        if ($validator->fails()) {
            return redirect()->route('appointments.create')
                        ->withErrors($validator)
                        ->withInput();
        }

        $validatedData = $validator->validated();

        try {
            $appointmentDateTime = Carbon::parse($validatedData['appointment_date'] . ' ' . $validatedData['appointment_time']);

            // Basic check for appointment conflicts (for the same practitioner at the same time)
            // This is a very basic check. Real-world would be more complex (e.g. practitioner availability slots)
            $existingAppointment = Appointment::where('practitioner_name', $validatedData['practitioner_name'])
                                    ->where('appointment_datetime', $appointmentDateTime)
                                    ->where('status', 'scheduled')
                                    ->exists();

            if ($existingAppointment) {
                return redirect()->route('appointments.create')
                            ->with('error', 'This time slot with ' . $validatedData['practitioner_name'] . ' is already booked. Please choose another time.')
                            ->withInput();
            }


            Appointment::create([
                'user_id' => Auth::id(),
                'practitioner_name' => $validatedData['practitioner_name'],
                'practitioner_type' => $validatedData['practitioner_type'],
                'appointment_datetime' => $appointmentDateTime,
                'reason' => $validatedData['reason'],
                'status' => 'scheduled',
            ]);

            return redirect()->route('appointments.create')->with('success', 'Appointment successfully scheduled!');

        } catch (\Exception $e) {
            \Illuminate\Support\Facades\Log::error('Appointment creation failed: ' . $e->getMessage());
            return redirect()->route('appointments.create')
                        ->with('error', 'Failed to schedule appointment. Please try again. Error: ' . $e->getMessage())
                        ->withInput();
        }
    }
}