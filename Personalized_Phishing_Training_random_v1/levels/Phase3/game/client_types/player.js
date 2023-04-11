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
                                return '<datalist id="tickmarks" style="font-size: 14px"><option value="0" label="0"></option><option value="100" label="100"></option></datalist><datalist id="tickmarks" style="font-size: 12px"><option value="0" label="Not Confident at All"></option><option value="100" label="Fully Confident"></option></datalist>Confidence Level: ' + value;
                            }},
                        min: 0,
                        max: 100,
                        step: 5, //this doesn't actually do anything, instead it is created in the cb function
                        initialValue: 0,
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
