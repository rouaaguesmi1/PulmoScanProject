<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class DicomViewerController extends Controller
{
    /**
     * Show the DICOM viewer page.
     */
    public function showViewer()
    {
        return view('dicom_viewer.viewer');
    }
}