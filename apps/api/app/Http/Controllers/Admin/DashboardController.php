<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Models\User;
use App\Models\VoiceModel;
use App\Models\UsageEvent;
use Illuminate\Support\Facades\DB;

class DashboardController extends Controller
{
    public function index()
    {
        // You may not have Eloquent models for all tables yet,
        // so for jobs_queue we can safely count via DB::table.
        $jobsCount = DB::table('jobs_queue')->count();

        return view('admin.dashboard', [
            'usersCount' => User::count(),
            'modelsCount' => VoiceModel::count(),
            'usageEventsCount' => UsageEvent::count(),
            'jobsCount' => $jobsCount,
        ]);
    }
}
