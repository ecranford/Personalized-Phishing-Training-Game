/**
 * # Game stages definition file
 * Copyright(c) 2022 Edward Cranford <cranford@cmu.edu>
 * MIT Licensed
 *
 * Stages are defined using the stager API
 *
 * http://www.nodegame.org
 * ---
 */

module.exports = function(treatmentName, settings, stager, setup, gameRoom) {
     stager

        .repeatStage('phase 2 content', 60)
        .step('phase 2')
        .step('feedback')

        .stage('phase 2 feedback');

};
