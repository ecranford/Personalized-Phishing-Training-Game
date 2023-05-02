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
        node.timer.random(2000).done();
    });

    stager.extendStep('terms and conditions', {
        cb: function() {
            node.done();
        }
    });

    stager.extendStep('consent form', {
        cb: function() {
            node.done();
        }
    });

    stager.extendStep('demographics', {
        cb: function() {
            node.done();
        }
    });

    stager.extendStep('experience survey', {
        cb: function() {
            node.done();
        }
    });

    stager.extendStep('instructions', {
        cb: function() {
            node.done();
        }
    });
    
    stager.extendStep('instruction quiz', {
        cb: function() {
            node.done();
        }
    });
    
    stager.extendStep('practice instructions', {
        cb: function() {
            node.done();
        }
    });

    stager.extendStep('practice description', {
        cb: function() {
            node.done();
        }
    });

    stager.extendStep('practice trial', {
        cb: function() {
            node.done();
        }
    });

    stager.extendStep('start game', {
        cb: function() {
            node.done();
        }
    });
    
    stager.extendStep('phase 1', {
        cb: function() {
            ///first trying to set values of the widget...but lastAppended is returning 'undefined'
            //node.widgets.lastAppended.setValues({ classification: {choice: 1, value: 'No', isCorrect: true}, confidence: {value: 0, noChange: false, isCorrect: true} });
            ///mabye instead...
            ///need to set message of done to edit 'data' object...
            ///so I can get data.forms.classification.choice and data.forms.confidence.value and times for each
            node.on.data('averages', function(msg) {
                let email_type = msg.data[0];
                console.log("Bot email type is: "+email_type);
                let avg_phish_acc = msg.data[1];
                console.log("Average Phish accuracy is: "+avg_phish_acc);
                let avg_ham_acc = msg.data[2];
                console.log("Average Ham accuracy is: "+avg_ham_acc);

                //set bot's choice based on average performance of all players...including bots
                let bot_choice;
                //set choice_value to 'yes' for PHISHING or 'no' for HAM
                let choice_value;

                if (email_type == 'PHISHING') {
                    if (Math.random() <= avg_phish_acc) {
                        bot_choice = 0;
                        choice_value = 'yes'
                    } else {
                        bot_choice = 1;
                        choice_value = 'no'
                    }
                } else {
                    if (Math.random() <= avg_ham_acc) {
                        bot_choice = 1;
                        choice_value = 'no'
                    } else {
                        bot_choice = 0;
                        choice_value = 'yes'
                    }
                };

                if (bot_choice == 0) {
                    console.log("Bot classification is: "+bot_choice+" "+choice_value+" (PHISHING)");
                } else {
                    console.log("Bot classification is: "+bot_choice+" "+choice_value+" (HAM)");
                }

                node.timer.random(5000).done({
                    order: [ 0, 1 ],
                    forms: { 
                        classification: {
                            id: 'classification',
                            choice: bot_choice,
                            time: 0,
                            nClicks: 1,
                            value: choice_value,
                            isCorrect: true,
                            attempts: []
                        },
                        confidence: {
                            value: 0,
                            noChange: false,
                            initialValue: 0,
                            totalMove: 0,
                            isCorrect: true,
                            time: 0
                        }},
                    missValues: [],
                    id: 'phase 2',
                    isCorrect: true
                });
            });
        }
    });

    stager.extendStep('phase 1 feedback', {
        cb: function() {
            node.done();
        }
    });

};
