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
        
        // Model permissions
        Permission::create(['name' => 'view_models']);
        Permission::create(['name' => 'upload_models']);
        Permission::create(['name' => 'update_own_models']);
        Permission::create(['name' => 'delete_own_models']);
        Permission::create(['name' => 'manage_all_models']);
        
        // Job permissions
        Permission::create(['name' => 'create_jobs']);
        Permission::create(['name' => 'view_own_jobs']);
        Permission::create(['name' => 'manage_all_jobs']);
        
        // User permissions
        Permission::create(['name' => 'manage_users']);
        Permission::create(['name' => 'view_all_users']);
        
        // System permissions
        Permission::create(['name' => 'view_stats']);
        Permission::create(['name' => 'manage_system']);
        
        // Training permissions (for future)
        Permission::create(['name' => 'train_models']);
        Permission::create(['name' => 'manage_training_queue']);

        // ======================================================================
        // Create Roles and Assign Permissions
        // ======================================================================
        
        // Guest role (for rate-limited public access)
        $guest = Role::create(['name' => 'guest']);
        $guest->givePermissionTo([
            'view_models',
        ]);

        // Regular User role
        $user = Role::create(['name' => 'user']);
        $user->givePermissionTo([
            'view_models',
            'create_jobs',
            'view_own_jobs',
        ]);

        // Premium User role (can upload models)
        $premium = Role::create(['name' => 'premium']);
        $premium->givePermissionTo([
            'view_models',
            'upload_models',
            'update_own_models',
            'delete_own_models',
            'create_jobs',
            'view_own_jobs',
        ]);

        // Creator role (can upload and train models)
        $creator = Role::create(['name' => 'creator']);
        $creator->givePermissionTo([
            'view_models',
            'upload_models',
            'update_own_models',
            'delete_own_models',
            'create_jobs',
            'view_own_jobs',
            'train_models',
        ]);

        // Moderator role
        $moderator = Role::create(['name' => 'moderator']);
        $moderator->givePermissionTo([
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
        ]);

        // Admin role (all permissions)
        $admin = Role::create(['name' => 'admin']);
        $admin->givePermissionTo(Permission::all());
    }
}
