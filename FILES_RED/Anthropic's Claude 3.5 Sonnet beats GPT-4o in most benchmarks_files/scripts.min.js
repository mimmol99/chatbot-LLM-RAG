/**
 * Popup Maker Popup Analytics v1.1
 */
 (function ($, document, undefined) {
    "use strict";

    $.fn.popmake.conversion_trigger = null;

    function trackForms(popup) {
        var $popup = $(popup),
            settings = PUM.getSettings(popup),
            $forms = $('form, .nf-form-cont', $popup);

        $forms.each(function () {
                var $form = $(this),
                    type;

                if ($('[name="popmake_pa_conversion_check"]', $form).length) {
                    return;
                }

                if ($form.hasClass('pum_sub_form')) {
                    type = 'pum';
                } else if ($form.find('.gform_body').length) {
                    type = 'gravityforms';
                } else if ($form.hasClass('ninja-forms-form')) {
                    type = 'ninjaforms_v2';
                } else if ($form.hasClass('nf-form-cont') || $form.parents('.nf-form-wrap').length) {
                    type = 'ninjaforms_v3';
                } else if ($form.hasClass('wpcf7-form')) {
                    type = 'contactform7';
                }

                switch (type) {
                case 'ninjaforms_v3':
                    if (typeof Marionette !== 'undefined' && typeof nfRadio !== "undefined") {
                        var formID = $form.attr('id').replace('nf-form-', '').replace('-cont', '');

                        $form.find('form').addClass('popmake_pa');

                        $popup
                            .on('pumAfterOpen', function () {
                                $form.find('form').addClass('popmake_pa');
                                nfRadio.channel('form-' + formID).request('add:extra', 'pum_popup_id', settings.id);
                                // nfRadio.channel('form-' + formID).request('add:extra', 'pum_open_event_id', '');
                            })
                            .on('pumOpenEventIdChanged', function (event, event_id) {
                                nfRadio.channel('form-' + formID).request('add:extra', 'pum_open_event_id', event_id.toString());
                            });
                    }

                    break;

                case 'pum':
                case 'contactform7':
                case 'ninjaforms_v2':
                case 'gravityforms':
                    $popup
                        .on('pumAfterOpen', function () {
                            $form.append('<input type="hidden" name="_popmake_pa_conversion_check" value="' + settings.id + '"/>');
                            $form.append('<input type="hidden" name="_popmake_pa_open_event_id" value=""/>');
                        })
                        .on('pumOpenEventIdChanged', function (event, event_id) {
                            $('[name="_popmake_pa_open_event_id"]', $form).val(event_id);
                        });

                    $form
                        .addClass('popmake_pa')
                        .bindFirst('submit', function () {
                            $.fn.popmake.conversion_trigger = $form.attr('id') || $.fn.popmake.utilities.getXPath($form);
                        });
                    break;
                }
            }
        );

    }

    function trackConversion() {
        $(document).trigger('pumConversion');
    }

    if ($.fn.bindFirst === undefined) {
        $.fn.bindFirst = function (which, handler) {
            var $el = $(this),
                events,
                registered;

            $el.unbind(which, handler);
            $el.bind(which, handler);

            events = $._data($el[0]).events;
            registered = events[which];
            registered.unshift(registered.pop());

            events[which] = registered;
        };
    }

    if ($.fn.outerHtml === undefined) {
        $.fn.outerHtml = function () {
            var $el = $(this).clone(),
                $temp = $('<div/>').append($el);

            return $temp.html();
        };
    }

    $(document)
        .on('pumConversion', function () {
            var $popup = PUM.getPopup($.fn.popmake.last_open_popup),
                ajaxData = {
                    'action': 'popmake_pa', // Calls our wp_ajax_nopriv_popmake_pa
                    'popup_id': $popup.data('popmake').id,
                    'event_type': 'conversion',
                    'open_event_id': $popup.data('open-event-id'),
                    'trigger': $.fn.popmake.conversion_trigger,
                    '_ajax_nonce': popmake_pa.nonce
                };

            $.ajax({
                type: 'POST',
                dataType: 'json',
                url: ajaxurl,
                async: false,
                data: ajaxData
            });
        });

    $('.pum')
        .on('pumInit', function () {
            var $popup = PUM.getPopup(this),
                settings = $popup.data('popmake'),
                convert_on = settings.meta.convert_on;

            trackForms(this);

            if (convert_on !== undefined && convert_on !== '') {
                switch (convert_on) {
                case 'form_submit':
                    $popup.find('form').filter(':not(.popmake_pa)').on('submit.popmake_pa', function () {
                        $.fn.popmake.conversion_trigger = $(this).attr('id') || 'Form Submit';
                        trackConversion();
                        $(this).off('submit.popmake_pa');
                    });
                    break;
                case 'button_click':
                    $popup.find('button, .button, .btn').on('click', function () {
                        console.log($(this));
                        $.fn.popmake.conversion_trigger = $(this).attr('id') || 'Button Click';
                        trackConversion();
                    });
                    break;
                case 'link_click':
                    $popup.find('a').on('click', function () {
                        $.fn.popmake.conversion_trigger = $(this).attr('id') || 'Link Click';
                        trackConversion();
                    });
                    break;
                }
            }

        })
        .on('pumBeforeOpen.analytics', function () {
            var $popup = PUM.getPopup(this),
                ajaxData = {
                    'action': 'popmake_pa', // Calls our wp_ajax_nopriv_popmake_pa
                    'popup_id': $popup.data('popmake').id,
                    'event_type': 'open',
                    'trigger': null,
                    '_ajax_nonce': popmake_pa.nonce
                },
                $last_trigger = null;

            try {
                $last_trigger = $($.fn.popmake.last_open_trigger);
            } catch (error) {
                $last_trigger = $();
            }

            if ($last_trigger.length) {
                ajaxData.trigger = $last_trigger.eq(0).outerHtml();
            } else if ($.fn.popmake.last_open_trigger !== null) {
                ajaxData.trigger = $.fn.popmake.last_open_trigger.toString();
            }

            $.ajax({
                type: 'POST',
                dataType: 'json',
                url: ajaxurl,
                data: ajaxData,
                success: function (data) {
                    if (data.success && data.event_id > 0) {
                        $popup.data('open-event-id', data.event_id);
                        $popup.trigger('pumOpenEventIdChanged', [data.event_id]);
                    } else {
                        $popup.data('open-event-id', null);
                    }
                }
            });

            //  TODO experiment with $(window).on('beforeunload' over $window.on('unload'
            $(window).one('unload.popmake_pa', function () {
                ajaxData = {
                    'action': 'popmake_pa', // Calls our wp_ajax_nopriv_popmake_pa
                    'popup_id': $popup.data('popmake').id,
                    'event_type': 'close',
                    'open_event_id': $popup.data('open-event-id'),
                    'trigger': 'Browser Closed',
                    '_ajax_nonce': popmake_pa.nonce
                };

                $.ajax({
                    type: 'POST',
                    dataType: 'json',
                    url: ajaxurl,
                    async: false,
                    data: ajaxData
                });
            });

        })
        .on('pumBeforeClose.analytics', function () {
            var $popup = PUM.getPopup(this),
                ajaxData = {
                    'action': 'popmake_pa', // Calls our wp_ajax_nopriv_popmake_pa
                    'popup_id': $popup.data('popmake').id,
                    'event_type': 'close',
                    'open_event_id': $popup.data('open-event-id'),
                    'trigger': $.fn.popmake.last_close_trigger,
                    '_ajax_nonce': popmake_pa.nonce
                };

            $(window).off('unload.popmake_pa');

            $.ajax({
                type: 'POST',
                dataType: 'json',
                url: ajaxurl,
                data: ajaxData
            });
        });
}(jQuery, document));
