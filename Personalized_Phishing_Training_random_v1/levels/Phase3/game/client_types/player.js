/**
 * # Player type implementation of the game stages
 * Copyright(c) 2022 Edward Cranford <cranford@cmu.edu>
 * MIT Licensed
 *
 * Each client type must extend / implement the stages defined in `game.stages`.
 * Upon connection each client is assigned a client type and it is automatically
 * setup with it.
 *
 * http://www.nodegame.org
 * ---
 */

"use strict";

module.exports = function(treatmentName, settings, stager, setup, gameRoom) {

    stager.setOnInit(function() {

        // Initialize the client.

        var header;

        // Setup page: header + frame.
        
        header = W.generateHeader();
        W.generateFrame();

        // Add widgets for header.
        this.visualStage = node.widgets.append('VisualStage', header, {rounds: false, addRound: false});
        this.visualRound = node.widgets.append('VisualRound', header, {texts: {
            round: 'Trial',
            step: 'Step',
            stage: 'Stage',
            roundLeft: 'Trials Left',
            stepLeft: 'Steps Left',
            stageLeft: 'Stages Left'
        }});
        this.visualTimer = node.widgets.append('VisualTimer', header);
        this.doneButton = node.widgets.append('DoneButton', header);

        // Additional debug information while developing the game.
        // this.debugInfo = node.widgets.append('DebugInfo', header)
    });

    stager.extendStep('phase 3', {
        frame: 'game.htm',
        init: function() {
            //node.game.visualTimer.hide();
            //W.setInnerHTML('phase', phase);
            //W.setInnerHTML('trial', trial);
        },
        donebutton: {
            text: 'Next'
        },

        cb: function() {
            node.on.data('phasedata', function(msg) {
                W.setInnerHTML('phase', msg.data[0]);
                W.setInnerHTML('trial', msg.data[1]);
            });
            W.cssRule('#container {max-width: 54em;}');
            W.cssRule('table, tr, td {border: 2px solid black;}');
            W.cssRule('table {width: 100%;}');
            W.cssRule('.panel {border: none;box-shadow: none;}');
            W.cssRule('.panel-body {padding: 0px;}');
            W.cssRule('table.choicetable {border: none;width: 50%;}');
            W.cssRule('.choicetable tr td {background: lightgray}');
            W.cssRule('.container-slider {height: 25px}');
            W.cssRule('.volume-slider::-moz-range-thumb {height: 25px; width: 25px; background: gray}');
            W.cssRule('.volume-slider::-moz-range-track {border: 1px solid gray}');
            node.on.data('email', function(msg) {
                W.setInnerHTML('source', msg.data[0].sender);
                W.setInnerHTML('subject', msg.data[0].subject);
                W.setInnerHTML('body', msg.data[0].body);
            });
            //set the step attribute to 5 so that the slider is easier to set...may revert to continuous values
            W.getElementsByClassName('volume-slider')[0].setAttribute('step', '5');
        },

        // Make a widget step for the classification questions.
        widget: {
            name: 'ChoiceManager', 
            root: 'classify',
            id: 'decision',
            options: {
                mainText: 'Answer the following questions:',
                forms: [
                    {
                        name: 'ChoiceTable',
                        id: 'classification',
                        orientation: 'H',
                        mainText: '1. Is this a phishing email?',
                        choices: ['Yes',
                                  'No'
                                ],
                        requiredChoice: true
                    },
                    {
                        name: 'Slider',
                        id: 'confidence',
                        mainText: '2. How confident are you in your answer for question 1?',
                        texts: {
                            currentValue: function(widget, value) {
                                return '<datalist id="tickmarks" style="font-size: 14px"><option value="50" label="50"></option><option value="100" label="100"></option></datalist><datalist id="tickmarks" style="font-size: 12px"><option value="50" label="Not Confident at All"></option><option value="100" label="Fully Confident"></option></datalist>Confidence Level: ' + value;
                            }},
                        min: 50,
                        max: 100,
                        step: 5, //this doesn't actually do anything, instead it is created in the cb function
                        initialValue: 50,
                        displayNoChange: false,
                        required: true
                    }
                ]
            }
        }
    });

    stager.extendStep('phase 3 feedback', {
        frame: 'feedback.htm',
        init: function() {
            node.game.visualTimer.show();
            node.game.doneButton.hide();
        },
        cb: function() {
            //node.timer.wait(10000).done();
            node.on.data('phasedata', function(msg) {
                W.setInnerHTML('phase', msg.data);
                W.setInnerHTML('phase2', msg.data);
            });
            node.on.data('scores', function(msg) {
                W.setInnerHTML('phase-score', msg.data[0]);
                W.setInnerHTML('total-score', msg.data[1]);
            });

            //W.cssRule('btn {text-align: center; display: inline-block; margin: 0 auto;}');
        },
        widget: {
            name: 'DoneButton',
            id: 'ResumeButton',
            text: 'Resume Now',
            className: 'btn btn-lg btn-secondary btn-block'
        },
        exit: function() {
            node.game.doneButton.show();
        }
    });
/*
    stager.extendStep('survey', {
        frame: 'survey.htm',  
        init: function() {
            node.game.visualTimer.hide();
        },      
        cb: function() {
            //nothing here yet, not sure what is needed
        },

        // Make a widget step for quick-version of experience survey.
        widget: {
            name: 'ChoiceManager',
            id: 'survey',
            options: {
                mainText: 'Please take a moment to answer the following questions. When you are finished, press the "DONE" button to end the experiment and receive your completion code.',
                forms: [
                    {
                        name: 'ChoiceTable',
                        id: 'survey1',
                        orientation: 'V',
                        mainText: '1. What is Phishing? (Select the most accurate description)',
                        choices: ['Pretending to be someone or a company to steal information or money',
                                  'Making a fake website that looks legitimate to steal information or money',
                                  'Sending spam or advertisement emails',
                                  'Tracking internet habits to send advertisements',
                                  'Hacking someone’s computer by trying different passwords',
                                  'Do not know'],
                        requiredChoice: true
                    },
                    {
                        name: 'ChoiceTable',
                        id: 'survey2',
                        orientation: 'V',
                        mainText: '2. By your estimate, what is the possible number of phishing emails you may have received in the last four months?',
                        choices: ['None',
                                  '1 to 2 phishing emails',
                                  '3 to 5 phishing emails',
                                  '5 to 10 phishing emails',
                                  'More than 10 phishing emails'],
                        requiredChoice: true
                    },
                    {
                        name: 'ChoiceTable',
                        id: 'survey3',
                        orientation: 'V',
                        mainText: '3. Did you receive training on phishing attacks in the recent past?',
                        choices: ['No',
                                  'Yes, I have undergone internet security awareness training',
                                  'I have read written educational material about phishing attacks and threats on the internet',
                                  'I was phished by my company as part of a training campaign and received feedback on how to detect phishing emails',
                                  'I have undergone formal computer security training or education'],
                        selectMultiple: true,
                        requiredChoice: true
                    }
                ],
                formsOptions: {
                    requiredChoice: true
                }
            }
        }
    });
*/
    stager.extendStep('end', {
        widget: {
            name: 'EndScreen',
            showEmailForm: false,
            feedback: {minChars: undefined}
        },
        init: function() {
            node.game.doneButton.destroy();
            node.game.visualTimer.destroy();
        }
    });
};
