/**
 * # Index script for nodeGame
 * Copyright(c) 2022 Edward Cranford <cranford@cmu.edu>
 * MIT Licensed
 *
 * http://nodegame.org
 * ---
 */
window.onload = function() {
    if ('undefined' === typeof node) {
        throw new Error('node is not loaded. Aborting.');
    }

    // All these properties will be overwritten
    // by remoteSetup from server.
    node.setup('nodegame', {
        verbosity: 100,
        debug : true,
        window : {
            promptOnleave : false
        },
        env : {
            auto : false,
            debug : false
        },
        events : {
            dumpEvents : true
        },
        socket : {
            type : 'SocketIo',
            reconnection : false
        }
    });
    // Connect to channel.
    // (If using an alias if default channel, must pass the channel name
    // as parameter to connect).
    node.connect('/Personalized_Phishing_Training_CogEV_v1');
};

window.onerror = function(msg, url, lineno, colno, error) {
    var str;

    // Modification.
    if (msg.indexOf('interrupted while the page was loading') !== -1) {
           location.reload();
           return;
     }

    msg = node.game.getCurrentGameStage().toString() +
        '@' + J.getTime() + '> ' +
        url + ' ' + lineno + ',' + colno + ': ' + msg;
    if (error) msg + ' - ' + JSON.stringify(error);
    that.lastError = msg;
    node.err(msg);
    if (node.debug) {
        W.init({ waitScreen: true });
        str = '<strong>DEBUG mode: client-side error ' +
              'detected.</strong><br/><br/>';
        str += msg;
        str += '</br></br>Open the DevTools in your browser ' +
        'for details.</br><em style="font-size: smaller">' +
        'This message will not be shown in production mode.</em>';
        W.lockScreen(str);
    }
    return !node.debug;
};