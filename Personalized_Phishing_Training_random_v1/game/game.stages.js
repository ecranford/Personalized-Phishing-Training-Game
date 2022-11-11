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
    // stages for PPT: consent, instructions, quiz, practice, pre-test, training, post-test, end, survey
     stager
        /*
        .stage('consent')
        .step('terms and conditions')
        .step('consent form')
        .step('demographics')
        .step('quick survey')

        .stage('game instructions')
        .step('instructions')
        .step('instruction quiz')

        .stage('practice instructions')
        .repeatStage('practice trials', 4)
        */
        .stage('start game')

        .repeatStage('phase 1', 20)

        .stage('phase 1 results')

        .stage('training')
        //.step('decision')
        //.step('feedback')

        //.stage('training-feedback')

        //.stage('post-test')
        //.step('decision')

        //.stage('post-test-feedback')
        
        //give feedback during 'end' stage
        .stage('end')

        //.stage('long-survey')

        .gameover();


    // Notice: here all stages have one step named after the stage.

    // Skip one stage.
    // stager.skip('instructions');

    // Skip multiple stages:
    // stager.skip([ 'instructions', 'quiz' ])

    // Skip a step within a stage:
    // stager.skip('stageName', 'stepName');

};
