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
    var emails = require("../../../../emails.json");
    //console.log(emails);

    // Must implement the stages here.

    stager.setOnInit(function() {
        // Initialize the client.
        // Will automatically save every entry in the database
        // to file memory.json (format ndjson).
        memory.stream();

        //if bot playing, need to create data object in registry
        this.pl.each(function(p) {
            if (channel.registry.getClient(p.id).data === undefined) {
                channel.registry.updateClient(p.id, { data: [] });
                console.log("Bot "+p.id+" created");
            };
            //console.log("player "+p.id+" data is: "+channel.registry.getClient(p.id).data);
        });
        

    });

    //choose which email to present on each trial of training (Phase 2)
    stager.extendStep('phase 2', {
        init: function() {
            let trial = node.game.getRound();
            let phase = node.game.getCurrentStepObj();
            console.log(phase.id[0].toUpperCase() + phase.id.substring(1));
            console.log("Trial "+trial);
            //for test phase, select 20% of players
            let ids = node.game.pl.id.getAllKeys();
            let bot_id = "1";
            while (ids.length < 10) {
                ids.unshift(bot_id);
                bot_id++;
              };
            console.log("Players List: "+ids);
            let num_players = Math.ceil(ids.length * 0.2); //change this as appropriate for game with 10 players (and dropouts)
            let selected_ids = [];
            // Randomly select n players without replacement
            do {
                selected_ids.unshift(ids.splice(Math.floor(Math.random() * ids.length), 1)[0]);
            }
            while (selected_ids.length < num_players);
            console.log("Selected players: "+selected_ids);

            let player_phish_accs = [];
            let player_ham_accs = [];

            //send average accuracies to bot
            //set bot's choice based on average performance of all players...including bots
            this.pl.each(function(p) {
                let player = channel.registry.getClient(p.id);
                //console.log(p.id+" data is:");
                //console.table(player.data);
                let total_phish_scores = player.data.filter(item => item.email_type === "PHISHING");
                //console.log(p.id+" phish accuracies is: "+total_phish_scores);
                let total_phish_score = 0;
                let average_phish_acc;
                if (total_phish_scores.length > 0) {
                    for(let i = 0; i < total_phish_scores.length; i++) {
                        total_phish_score += total_phish_scores[i].accuracy;
                    }
                    average_phish_acc = total_phish_score / total_phish_scores.length;
                    player_phish_accs.push(average_phish_acc);
                } else {
                    average_phish_acc = null;
                }
                //console.log(p.id+" avg phish accuracy is: "+average_phish_acc);
                
                let total_ham_scores = player.data.filter(item => item.email_type === "HAM");
                //console.log(p.id+" ham accuracies is: "+total_ham_scores);
                let total_ham_score = 0;
                let average_ham_acc;
                if (total_ham_scores.length > 0) {
                    for(let i = 0; i < total_ham_scores.length; i++) {
                        total_ham_score += total_ham_scores[i].accuracy;
                    }
                    average_ham_acc = total_ham_score / total_ham_scores.length;
                    player_ham_accs.push(average_ham_acc);
                } else {
                    average_ham_acc = null;
                }
                //console.log(p.id+" avg ham accuracy is: "+average_ham_acc);
            });
            //console.log("Phish accuracies is: "+player_phish_accs);
            //console.log("Ham accuracies is: "+player_ham_accs);

            let avg_phish_acc;
            let avg_ham_acc;
            if (player_phish_accs.length > 0) {
                avg_phish_acc = player_phish_accs.reduce((a, b) => a + b) / player_phish_accs.length;
            } else {
                avg_phish_acc = 0.5;
            };
            if (player_ham_accs.length > 0) {
                avg_ham_acc = player_ham_accs.reduce((a, b) => a + b) / player_ham_accs.length;
            } else {
                avg_ham_acc = 0.5;
            };
            //console.log("Avg phish accuracy is: "+avg_phish_acc);
            //console.log("Avg ham accuracy is: "+avg_ham_acc);   

            //send phase, trial, and email data to each participant
            this.pl.each(function(p) {
                let player = channel.registry.getClient(p.id);
                //change this selection for CogModel and RMAB versions so that it requests and gets info from model.
                let seen_emails_list = player.data.filter(item => item.email_id);
                let seen_emails = [];
                for (let i = 0; i < seen_emails_list.length; i++) {
                    //console.log(scores[i].accuracy);
                    seen_emails.push(seen_emails_list[i].email_id);
                };
                console.log(p.id+" Seen emails: "+seen_emails);
                let email_num;
                //for test phase, send phishing email to 20% of players
                if (selected_ids.includes(p.id)) {
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

                node.say("phasedata", p.id, [phase.id[0].toUpperCase() + phase.id.substring(1), trial]);
                var email = emails.filter(el => {return el['id'] === email_num.toString();});
                console.log(p.id+" Email ID: "+email[0].id);
                p.email_id = email[0].id;
                p.email_type = email[0].type;
                node.say("email", p.id, email);                                    
                node.say("averages", p.id, [p.email_type, avg_phish_acc, avg_ham_acc]);   
                //console.log(p);
            });
            
        },
        cb: function () {
            node.on.done(function(msg) {
                //console.log(msg);
                let data = msg.data;
                let player = channel.registry.getClient(data.player);
                //console.log(data);
                let acc;
                let choice = data.forms.classification.choice;
                
                let email_id = this.pl.get(data.player).email_id;
                let email_type = this.pl.get(data.player).email_type;
                let classification;
                let confidence_val = data.forms.confidence.value;
                //algorithm for 50-100 scale
                //((0.01*data.forms.confidence.value)*50)+50;

                switch (choice) {
                    case 0:
                        if (email_type == 'PHISHING') {acc = 1.0;} else {acc = 0.0;};
                        classification = 'PHISHING';
                        break;
                    case 1:
                        if (email_type == 'HAM') {acc = 1.0;} else {acc = 0.0;};
                        classification = 'HAM';
                        break;
                    default:
                        acc = 0.0;
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
            });
        }
    });

    // present feedback during training
    stager.extendStep('feedback', {
        init: function() {
            let trial = node.game.getRound();
            //console.log("feedback for trial: "+trial);
            this.pl.each(function(p) {
                //console.log("Player: "+p.id);
                let player = channel.registry.getClient(p.id);
                //console.log(player.data);
                let acc = player.data.filter(item => item.phase === "phase 2" && item.trial === trial)[0].accuracy;
                console.log(p.id+" Acc: "+acc);
                let email_type = player.data.filter(item => item.phase === "phase 2" && item.trial === trial)[0].email_type;
                console.log(p.id+" Email Type: "+email_type);
                if (acc == 0 && email_type == 'PHISHING') {
                    console.log("Incorrect phishing classification. Giving feedback to: "+p.id);
                    node.say("feedback", p.id, 0);
                } else {
                    console.log("No feedback given to "+p.id+" for ham emails or correct phishing classification.");
                    node.say("feedback", p.id, 1);
                }
            });
        }
    });

    // Phase 2 feedback
    stager.extendStep('phase 2 feedback', {
        init: function() {
            let prev_phase = node.game.getPreviousStep();
            let phase = node.game.getStepId(prev_phase);
            let phase_name = phase[0].toUpperCase() + phase.substring(1);
            //send phase and score data to each participant
            this.pl.each(function(p) {
                let player = channel.registry.getClient(p.id);
                //phase data
                node.say("phasedata", p.id, phase_name);
                //phase score data
                let phase_scores = player.data.filter(item => item.phase === "phase 2");
                let phase_score = 0;
                for (let i = 0; i < phase_scores.length; i++) {
                    //console.log(scores[i].accuracy);
                    phase_score += phase_scores[i].accuracy;
                };
                //total score data
                let total_scores = player.data.filter(item => item.accuracy);
                let total_score = 0;
                for (let i = 0; i < total_scores.length; i++) {
                    //console.log(scores[i].accuracy);
                    total_score += total_scores[i].accuracy;
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
                let levelName = 'Phase3';
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
