<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;

class AssetsAdminController extends Controller
{
    /**
     * Display the assets page.
     */
    public function index()
    {
        return view('admin.assets.index');
    }
}
