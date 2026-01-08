<?php

return [
    // Admin UI domain. Used by routes/web.php
    'domain' => env('ADMIN_DOMAIN', 'admin.morphvox.net'),
    
    // Main app domain (used for routing the global login redirect)
    'main_domain' => env('MAIN_DOMAIN', 'morphvox.net'),
];
