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

        .repeatStage('phase 3', 20)

        .stage('phase 3 feedback')

        //.stage('survey')
        
        //give feedback during 'end' stage
        .stage('end')

        .gameover();

};
