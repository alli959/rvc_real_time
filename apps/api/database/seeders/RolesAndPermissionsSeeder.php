<?php

namespace Database\Seeders;

use Illuminate\Database\Seeder;
use Spatie\Permission\Models\Role;
use Spatie\Permission\Models\Permission;

class RolesAndPermissionsSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        // Reset cached roles and permissions
        app()[\Spatie\Permission\PermissionRegistrar::class]->forgetCachedPermissions();

        // ======================================================================
        // Create Permissions
        // ======================================================================
        $permissions = [
            // Model permissions
            'view_models',
            'upload_models',
            'update_own_models',
            'delete_own_models',
            'manage_all_models',
            // Job permissions
            'create_jobs',
            'view_own_jobs',
            'manage_all_jobs',
            // User permissions
            'manage_users',
            'view_all_users',
            // System permissions
            'view_stats',
            'manage_system',
            // Training permissions (for future)
            'train_models',
            'manage_training_queue',
        ];

        foreach ($permissions as $permission) {
            Permission::firstOrCreate(['name' => $permission]);
        }

        // ======================================================================
        // Create Roles and Assign Permissions
        // ======================================================================
        $roles = [
            'guest' => ['view_models'],
            'user' => [
                'view_models',
                'create_jobs',
                'view_own_jobs',
            ],
            'premium' => [
                'view_models',
                'upload_models',
                'update_own_models',
                'delete_own_models',
                'create_jobs',
                'view_own_jobs',
            ],
            'creator' => [
                'view_models',
                'upload_models',
                'update_own_models',
                'delete_own_models',
                'create_jobs',
                'view_own_jobs',
                'train_models',
            ],
            'moderator' => [
                'view_models',
                'upload_models',
                'update_own_models',
                'delete_own_models',
                'manage_all_models',
                'create_jobs',
                'view_own_jobs',
                'manage_all_jobs',
                'view_all_users',
                'view_stats',
            ],
            'admin' => Permission::all()->pluck('name')->toArray(),
        ];

        foreach ($roles as $roleName => $rolePermissions) {
            $role = Role::firstOrCreate(['name' => $roleName]);
            $role->syncPermissions($rolePermissions);
        }
    }
}
