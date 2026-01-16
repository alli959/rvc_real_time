<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;

class MetricsAdminController extends Controller
{
    /**
     * Display the metrics page.
     */
    public function index()
    {
        return view('admin.metrics.index');
    }
}
