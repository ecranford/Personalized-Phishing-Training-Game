/**
 * # Bot type implementation of the game stages
 * Copyright(c) 2022 Edward Cranford <cranford@cmu.edu>
 * MIT Licensed
 *
 * http://www.nodegame.org
 * ---
 */

module.exports = function(treatmentName, settings, stager,
                          setup, gameRoom, node) {

    stager.setDefaultCallback(function() {
        node.timer.random.done();
    });

    stager.extendStep('game', {
        roles: {
            DICTATOR: {
                cb: function() {
                    // This Dictator BOT makes an initial offer of 50
                    // or repeats any offer previously received.
                    node.timer.random(3000).done({
                        offer: 'undefined' === typeof node.game.offer ?
                                50 : node.game.offer
                    });
                }
            },
            OBSERVER: {
                cb: function() {
                    node.on.data('decision', function(msg) {
                        // Store last offer.
                        node.game.offer = msg.data;
                        node.timer.random(3000).done();
                    });
                }
            }
        }
    });
};
