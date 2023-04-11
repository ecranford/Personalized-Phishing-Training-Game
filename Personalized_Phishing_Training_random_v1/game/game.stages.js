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
     
     .stage('consent')
     .step('terms and conditions')
     .step('consent form')
     .step('demographics')
     .step('experience survey')

     .stage('game instructions')
     .step('instructions')
     .step('instruction quiz')

     .stage('practice instructions')

     .repeatStage('practice trials', 4)
     
     .stage('start game')

     .repeatStage('phase 1', 20)

     .stage('phase 1 feedback')
     
     .gameover();

};
