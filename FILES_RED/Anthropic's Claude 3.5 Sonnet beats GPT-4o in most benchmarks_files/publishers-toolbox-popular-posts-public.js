(function ($) {
    'use strict';

    /**
     * Track views with ajax.
     */
    jQuery(document).ready(function () {
        var body = jQuery('body');

        /**
         * Track a page view.
         */
        setTimeout(function () {
            trackView('view');
        }, 3000);

        /**
         * Track a page read.
         */
        setTimeout(function () {
            trackView('read');
        }, body.data('interval'));

        function trackView(setupActive) {
            jQuery.post(ptxFrontendObject.ajax_url, {
                action: 'ptx_frontend_action',
                security: ptxFrontendObject._nonce,
                post_id: ptxFrontendObject.post_id,
                log_type: setupActive,
            });
        }
    });
})(jQuery);
