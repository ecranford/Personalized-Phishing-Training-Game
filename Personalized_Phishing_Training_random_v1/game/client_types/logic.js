/**
 * # Logic type implementation of the game stages
 * Copyright(c) 2022 Edward Cranford <cranford@cmu.edu>
 * MIT Licensed
 *
 * http://www.nodegame.org
 * ---
 */

"use strict";

const ngc = require('nodegame-client');
const J = ngc.JSUS;

module.exports = function(treatmentName, settings, stager, setup, gameRoom) {

    let node = gameRoom.node;
    let channel = gameRoom.channel;
    let memory = node.game.memory;
    //Load database of emails
    var emails = require("../../emails.json");
    //console.log(emails);

    // Must implement the stages here.

    stager.setOnInit(function() {
        // Initialize the client.
        // Will automatically save every entry in the database
        // to file memory.json (format ndjson).
        memory.stream();

        //create data object for each player in registry to use across levels
        this.pl.each(function(p) {
        channel.registry.updateClient(p.id, { data: [] });
        });

    });

    //choose which email to present on each trial of practice
    stager.extendStep('practice trials', {
        cb: function() {
        this.pl.each(function(p) {
                var email = emails.filter(el => {return el['id'] === "ex"+node.game.getRound();});
                node.say("email", p.id, email);
            });
        }
    });

    //choose which email to present on each trial of pre-test (Phase 1)
    stager.extendStep('phase 1', {
        init: function() {
            let trial = node.game.getRound();
            let phase = node.game.getCurrentStepObj();
            console.log(phase.id[0].toUpperCase() + phase.id.substring(1));
            console.log("Trial "+trial);
            //send phase, trial, and email data to each participant
            this.pl.each(function(p) {
                let player = channel.registry.getClient(p.id);
                //change this selection for CogModel and RMAB versions so that it requests and gets info from model.
                let seen_emails_list = player.data.filter(item => item.email_id);
                //console.log("Seen emails list from registry: "+player.data.filter(item => item.email_id));
                let seen_emails = [];
                let i;
                for (i = 0; i < seen_emails_list.length; i++) {
                    //console.log(scores[i].accuracy);
                    seen_emails.push(seen_emails_list[i].email_id);
                };
                console.log(p.id+" Seen emails: "+seen_emails);
                let seen_email_types_list = player.data.filter(item => item.email_type && item.phase === "phase 1");
                let seen_email_types = [];
                for (i = 0; i < seen_email_types_list.length; i++) {
                    //console.log(scores[i].accuracy);
                    seen_email_types.push(seen_email_types_list[i].email_type);
                };
                console.log(p.id+" Seen email types: "+seen_email_types);
                let num_phishing = seen_email_types.filter(x => x=="PHISHING").length;
                console.log(p.id+" Number phishing: "+num_phishing);
                let num_ham = seen_email_types.filter(x => x=="HAM").length;
                console.log(p.id+" Number ham: "+num_ham);
                let selected_email_type;
                let email_num;
                //for pre- and post-test phases, send random 10 phishing and 10 ham
                if (num_phishing <= 9) {
                    if (num_ham <= 9) {
                        if (Math.floor(Math.random() * 2) == 0) {
                            selected_email_type = "PHISHING";
                            console.log("Both <= 9, sending phishing to "+p.id);
                        } else {
                            selected_email_type = "HAM";
                            console.log("Both <= 9, sending ham to "+p.id);
                        };
                    } else {
                        selected_email_type = "PHISHING";
                        console.log("10 ham, sending phishing to "+p.id);
                    };
                  } else {
                    selected_email_type = "HAM";
                    console.log("10 phishing, sending ham to "+p.id);
                };
                if (selected_email_type == "PHISHING") {
                    do {
                        email_num = Math.floor(Math.random() * 188) + 1;
                    }
                    while (seen_emails.includes(''+email_num));
                    
                } else {
                    do {
                        email_num = Math.floor(Math.random() * 177) + 189;
                    }
                    while (seen_emails.includes(''+email_num));
                }; 
                console.log("Getting email number "+email_num+" for "+p.id);

                node.say("phasedata", p.id, [phase.id[0].toUpperCase() + phase.id.substring(1), trial]);
                var email = emails.filter(el => {return el['id'] === email_num.toString();});
                console.log("Verifying Email ID "+email[0].id+" for "+p.id);
                //console.log("Email sender: "+email[0].sender);
                //console.log("Email subject: "+email[0].subject);
                //console.log("Email body: "+email[0].body);
                p.email_id = email[0].id;
                p.email_type = email[0].type;
                node.say("email", p.id, email);

                //console.log(p);
            });
            
        },
        cb: function () {
            node.once.done(function(msg) {
                let data = msg.data;
                let player = channel.registry.getClient(data.player);
                //console.log(data);
                let acc;
                let choice = data.forms.classification.choice;
                
                let email_id = this.pl.get(data.player).email_id;
                let email_type = this.pl.get(data.player).email_type;
                let classification;
                let confidence_val = ((0.01*data.forms.confidence.value)*50)+50;

                switch (choice) {
                    case 0:
                        if (email_type == 'PHISHING') {acc = 1;} else {acc = 0;};
                        classification = 'PHISHING';
                        break;
                    case 1:
                        if (email_type == 'HAM') {acc = 1;} else {acc = 0;};
                        classification = 'HAM';
                        break;
                    default:
                        acc = 0;
                        classification = 'HAM';
                };
                console.log(data.player+" Classification: "+choice+" "+classification);
                console.log(data.player+" Confidence: "+confidence_val);
                console.log(data.player+" Accuracy: "+acc);
                console.log("------------------------");

                //save data to memory
                node.game.memory.add({recordType: "decision",
                    player: data.player,
                    session: data.session,
                    stage: data.stage,
                    time: data.time,
                    timestamp: data.timestamp,
                    phase: data.stepId,
                    trial: data.stage.round,
                    email_id: email_id,
                    email_type: email_type,
                    class_val: choice,
                    classification: classification,
                    class_time: data.forms.classification.time,
                    confidence: confidence_val,
                    conf_time: data.forms.confidence.time,
                    accuracy: acc
                });
                
                //save data to registry
                player.data.push({
                    player: data.player,
                    session: data.session,
                    stage: data.stage,
                    time: data.time,
                    timestamp: data.timestamp,
                    phase: data.stepId,
                    trial: data.stage.round,
                    email_id: email_id,
                    email_type: email_type,
                    class_val: choice,
                    classification: classification,
                    class_time: data.forms.classification.time,
                    confidence: confidence_val,
                    conf_time: data.forms.confidence.time,
                    accuracy: acc   
                });

                //console.log(player.data);
                
            });
        }
    });

    // Phase 1 feedback
    stager.extendStep('phase 1 feedback', {
        init: function() {
            let prev_phase = node.game.getPreviousStep();
            let phase = node.game.getStageId(prev_phase);
            let phase_name = phase[0].toUpperCase() + phase.substring(1);
            //send phase and score data to each participant
            this.pl.each(function(p) {
                let player = channel.registry.getClient(p.id);
                //phase data
                node.say("phasedata", p.id, phase_name);
                //phase score data
                let phase_scores = player.data.filter(item => item.accuracy && item.phase === "phase 1");
                let phase_score = 0;
                let i;
                for (i = 0; i < phase_scores.length; i++) {
                    //console.log(scores[i].accuracy);
                    phase_score += phase_scores[i].accuracy;
                };
                //total score data
                let total_scores = player.data.filter(item => item.accuracy);
                let total_score = 0;
                let j;
                for (j = 0; j < total_scores.length; j++) {
                    //console.log(scores[i].accuracy);
                    total_score += total_scores[j].accuracy;
                }
                console.log(p.id+" "+phase_name+" Score: "+phase_score);
                console.log(p.id+" Total Score: "+total_score);
                node.say("scores", p.id, [phase_score,total_score]);
                // Update earnings counts, so that it can be saved
                // with GameRoom.computeBonus.
                gameRoom.updateWin(p.id, phase_score);

            });
        },
        cb: function() {
            node.on.data('level_done', function(msg) {
                // currentRoom is optional, avoid lookup.
                let currentRoom; // let currentRoom = gameRoom.name; 
                let levelName = 'Phase2';
                // Move client to the next level.
                // (async so that it finishes all current step operations).
                setTimeout(function() {
                    console.log('moving client to next level: ', msg.from);
                    channel.moveClientToGameLevel(msg.from, levelName, currentRoom);
                }, 100);
            });
        }
    });

};
