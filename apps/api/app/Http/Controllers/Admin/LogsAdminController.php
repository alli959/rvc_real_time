<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;

class LogsAdminController extends Controller
{
    /**
     * Display the logs page.
     */
    public function index()
    {
        return view('admin.logs.index');
    }
}
