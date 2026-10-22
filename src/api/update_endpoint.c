#include "api/update_endpoint.h"
#include "api/json_write.h"
#include "update/update_check.h"
#include "update/release_links.h"

char *api_update_json(int force) {
    int started = update_service_poll(force);
    UpdateInfo ui;
    UpdateOptions uo;
    update_service_get(&ui);
    int have_opts = update_service_options(&uo);
    JsonBuf jb;
    jb_init(&jb);
    jb_raw(&jb, "{"); jb_key(&jb, "status");     jb_str(&jb, update_status_name(ui.status));
    jb_raw(&jb, ","); jb_key(&jb, "current");    jb_str(&jb, ui.current);
    jb_raw(&jb, ","); jb_key(&jb, "latest");     jb_str(&jb, ui.latest[0] ? ui.latest : NULL);
    jb_raw(&jb, ","); jb_key(&jb, "url");        jb_str(&jb, ui.url[0] ? ui.url : NULL);
    jb_raw(&jb, ","); jb_key(&jb, "published");  jb_str(&jb, ui.published[0] ? ui.published : NULL);
    jb_raw(&jb, ","); jb_key(&jb, "prerelease"); jb_bool(&jb, ui.latest_prerelease);
    jb_raw(&jb, ","); jb_key(&jb, "checked_at"); jb_int(&jb, ui.checked_at);
    jb_raw(&jb, ","); jb_key(&jb, "error");      jb_str(&jb, ui.error[0] ? ui.error : NULL);
    jb_raw(&jb, ","); jb_key(&jb, "checking");   jb_bool(&jb, started);
    /* What changed in the running build: its release notes (release builds
     * only) and the changelog at that tag (else the default branch). */
    char notes_url[320], changelog_url[320];
    int have_notes = have_opts && release_notes_url(uo.api_base, uo.repo, ui.current,
                                                    notes_url, sizeof(notes_url)) == 0;
    int have_log = have_opts && release_changelog_url(uo.api_base, uo.repo, ui.current,
                                                      changelog_url, sizeof(changelog_url)) == 0;
    jb_raw(&jb, ","); jb_key(&jb, "release_notes_url"); jb_str(&jb, have_notes ? notes_url : NULL);
    jb_raw(&jb, ","); jb_key(&jb, "changelog_url");     jb_str(&jb, have_log ? changelog_url : NULL);
    /* Server-side settings, so the UI can describe what is checked and how often. */
    jb_raw(&jb, ","); jb_key(&jb, "settings"); jb_raw(&jb, "{");
    jb_key(&jb, "enabled");        jb_bool(&jb, have_opts && uo.enabled && uo.interval_hours > 0 &&
                                               ui.status != UPDATE_STATUS_DISABLED);
    jb_raw(&jb, ","); jb_key(&jb, "interval_hours"); jb_int(&jb, have_opts ? uo.interval_hours : 0);
    jb_raw(&jb, ","); jb_key(&jb, "channel"); jb_str(&jb, update_channel_name(have_opts ? uo.channel : 0));
    jb_raw(&jb, ","); jb_key(&jb, "repo");    jb_str(&jb, have_opts ? uo.repo : NULL);
    jb_raw(&jb, ","); jb_key(&jb, "force_min_interval_s"); jb_int(&jb, have_opts ? uo.force_min_interval_s : 0);
    jb_raw(&jb, "}}");
    return jb_finish(&jb);
}
