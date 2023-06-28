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

//get packages to write data.csv file at end of phase 3
var stringify = require('csv-stringify/sync');
var fs = require('fs');

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

        node.on.pdisconnect(function(player) {
            //console.log(player);
            if (player.disconnected && player.stage.stage < 3) {
                //don't allow player to reconnect if kicked
                player.allowReconnect = false;

                //Saves bonus file, and notifies players.
                console.log("Saving data for "+player.id);
                gameRoom.computeBonus({say: false, amt: true, addDisconnected: true, append: true, clients: [player.id], backup: false});
                // Save times of all stages in case need to figure out how much base pay to pay them
                memory.select('player', '=', player.id).save(player.id+'_times.csv', {
                    header: [
                        'session', 'player', 'stage', 'step', 'round', 'timestamp',
                        'time', 'timeup'
                    ]
                });
                // Save current level data to csv
                memory.select('recordType', '=', 'decision').and('player', '=', player.id).save(player.id+'_data.csv', {
                    header: [
                        'session', 'group', 'player', 'WorkerId', 'mturkid','type',
                        'stage', 'step', 'round', 'time', 'timestamp','phase','trial',
                        'email_id', 'email_type','class_val','classification',
                        'class_time','confidence','conf_time','accuracy'
                    ]
                });
                
                //save all data to main data.csv
                //modify group id for all stages to match group id of Phase 2
                for(let i = 0; i < player.data.length; i++) {
                    player.data[i].group = player.group;
                };

                //write player.data to a data.csv file (if exists, append)
                if (fs.existsSync('./games_available/Personalized_Phishing_Training_random_v1/data/data.csv')) {
                    let output = stringify.stringify(player.data,{header: false});
                    try {
                        fs.appendFileSync('./games_available/Personalized_Phishing_Training_random_v1/data/data.csv', output, 'utf8');
                        console.log('Data saved for player '+player.id);
                    } catch (err) {
                        console.log('Some error occured - file either not saved or corrupted file saved for player '+player.id);
                    };
                } else {
                    let output = stringify.stringify(player.data,{header: true});
                    try {
                        fs.writeFileSync('./games_available/Personalized_Phishing_Training_random_v1/data/data.csv', output);
                        console.log('Data saved for player '+player.id);
                    } catch (err) {
                        console.log('Some error occured - file either not saved or corrupted file saved for player '+player.id);
                    };
                };

                let totalTime = Date.now()-player.startTime;
                let bonus = [{
                    id: player.id,
                    type: player.clientType,
                    workerid: player.WorkerId,
                    hitid: player.HITId,
                    assignmentid: player.AssignmentId,
                    access: 'NA',
                    exit: player.ExitCode,
                    totaltime: totalTime,
                    approve: 1,
                    reject: 0,
                    basepay: node.game.settings.BASE_PAY,
                    bonus: player.winRaw*node.game.settings.EXCHANGE_RATE,
                    totalpay: (totalTime/60000)*0.10, //time in minutes * $0.10 per minute, no bonus
                    disconnected: player.disconnected == null ? 0 : player.disconnected,
                    disconnectStage: '3.'+player.disconnectedStage.stage+'.'+player.disconnectedStage.step+'.'+player.disconnectedStage.round
                }];
                // Save bonus info to main bonus.csv
                if (fs.existsSync('./games_available/Personalized_Phishing_Training_random_v1/data/bonus.csv')) {
                    let output = stringify.stringify(bonus,{header: false});
                    try {
                        fs.appendFileSync('./games_available/Personalized_Phishing_Training_random_v1/data/bonus.csv', output, 'utf8');
                        console.log('Bonus data saved for player '+player.id);
                    } catch (err) {
                        console.log('Some error occured - file either not saved or corrupted file saved for player '+player.id);
                    };
                } else {
                    let output = stringify.stringify(bonus,{header: true});
                    try {
                        fs.writeFileSync('./games_available/Personalized_Phishing_Training_random_v1/data/bonus.csv', output, 'utf8');
                        console.log('Bonus data saved for player '+player.id);
                    } catch (err) {
                        console.log('Some error occured - file either not saved or corrupted file saved for player '+player.id);
                    };
                };

                //console.log(node.game.pl.id.getAllKeys());
                node.game.stop();
            };
        });

        console.log("----------Beginning Phase 3----------");
    });

    stager.extendStep('phase 3', {
        init: function() {
            //moved everything to the cb function and added a 500 ms wait
        },
        cb: function () {
            node.timer.wait(500).exec(function () { 
                let trial = node.game.getRound();
                let phase = node.game.getCurrentStepObj();
                console.log("------------------------");
                console.log(phase.id[0].toUpperCase() + phase.id.substring(1));
                console.log("Trial "+trial);
                //choose which email to present on each trial of post-test (Phase 3)
                this.pl.each(function(p) {
                    let player = channel.registry.getClient(p.id);
                    //change this selection for CogModel and RMAB versions so that it requests and gets info from model.
                    let seen_emails_list = player.data.filter(item => item.email_id);
                    let seen_emails = [];
                    for (let i = 0; i < seen_emails_list.length; i++) {
                        //console.log(scores[i].accuracy);
                        seen_emails.push(seen_emails_list[i].email_id);
                    };
                    console.log(p.id+" Seen emails has duplicates: "+((new Set(seen_emails)).size !== seen_emails.length));
                    let seen_email_types_list = player.data.filter(item => item.phase === "phase 3");
                    let seen_email_types = [];
                    for (let i = 0; i < seen_email_types_list.length; i++) {
                        //console.log(scores[i].accuracy);
                        seen_email_types.push(seen_email_types_list[i].email_type);
                    };
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
                    console.log("Fetching email number "+email_num+" for "+p.id);

                    //send phase, trial, and email data to each participant
                    node.say("phasedata", p.id, [phase.id[0].toUpperCase() + phase.id.substring(1), trial]);
                    var email = emails.filter(el => {return el['id'] === email_num.toString();});
                    if (email_num.toString() !== email[0].id) {
                        console.log("Error: wrong email retrieved from database for "+p.id);
                    }
                    //console.log("Verifying Email ID "+email[0].id+" for "+p.id);
                    p.email_id = email[0].id;
                    p.email_type = email[0].type;
                    node.say("email", p.id, email);
                });
            });

            node.on.done(function(msg) {
                let data = msg.data;

                let player = channel.registry.getClient(data.player);

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

                //save data to file
                memory.add({
                    recordType: "decision",
                    player: data.player,
                    WorkerId: player.WorkerId,
                    mturkid: player.mturkid,
                    type: player.clientType,
                    session: data.session,
                    group: player.group,
                    stage: data.stage,
                    time: data.time,
                    timestamp: data.timestamp,
                    timeup: +data.timeup,
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
                    WorkerId: player.WorkerId,
                    mturkid: player.mturkid,
                    type: player.clientType,
                    session: data.session,
                    group: player.group,
                    stage: data.stage.stage,
                    step: data.stage.step,
                    round: data.stage.round,
                    time: data.time,
                    timestamp: data.timestamp,
                    timeup: +data.timeup,
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

                //track timeouts in player object
                if (typeof player.total_timeout === 'number') {
                    if (data.timeup && typeof choice != 'number') {
                        player.total_timeout = player.total_timeout + 1;
                        player.consec_timeout = player.consec_timeout + 1;
                    } else {
                        player.consec_timeout = 0;
                    };
                } else {
                    if (typeof choice === 'number') {
                        player.total_timeout = 0;
                        player.consec_timeout = 0;
                    } else {
                        player.total_timeout = +data.timeup;
                        player.consec_timeout = +data.timeup; 
                    }
                };
                console.log(data.player+" Total timeouts: "+player.total_timeout);
                console.log(data.player+" Consecutive timeouts: "+player.consec_timeout);

                //disconnect if 3 consecutive timeouts or 5 total timeouts
                if (player.total_timeout >= 5 || player.consec_timeout >= 3) {
                    //disconnect player here
                    //console.log(node.game.pl.id.getAllKeys());
                    console.log(data.player+" was removed");
                    //Redirect player to disconnected page?
                    node.redirect('disconnected.htm', player.id);
                    //node.redirect('disconnected.htm'+'?code='+player.ExitCode, player.id);
                    
                    //alternative method to disconnecte a player
                    //node.remoteAlert('You have been disconnected due to inactivity.\nPlease return to the HIT and enter this code to receive your partial payment: ', player.id);
                    //node.disconnectClient(player);
                    
                    //console.log(node.game.pl.id.getAllKeys());
                    node.game.pause();
                    //node.game.resume();
                    //node.game.gameover();
                };
            });
        }
    });

    // Phase 3 feedback
    stager.extendStep('phase 3 feedback', {
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
                let phase_scores = player.data.filter(item => item.phase === "phase 3");
                let phase_score = 0;
                for (let i = 0; i < phase_scores.length; i++) {
                    //console.log(scores[i].accuracy);
                    phase_score += phase_scores[i].accuracy;
                };
                //total score data
                let total_score = 0;
                for (let i = 0; i < player.data.length; i++) {
                    //console.log(scores[i].accuracy);
                    total_score += player.data[i].accuracy;
                }
                console.log(p.id+" "+phase_name+" Score: "+phase_score);
                console.log(p.id+" Total Score: "+total_score);
                node.say("scores", p.id, [phase_score,total_score]);
                // Update earnings counts, so that it can be saved
                // with GameRoom.computeBonus.
                gameRoom.updateWin(p.id, phase_score);

            });
        }
    });

    stager.extendStep('end', {
        init: function() {

            // Feedback.
            memory.view('feedback').stream({
                header: [ 'time', 'timestamp', 'player', 'feedback' ],
                format: 'csv'
            });

            // Email.
            memory.view('email').stream({
                header: [ 'timestamp', 'player', 'email' ],
                format: 'csv'
            });

        },
        cb: function() {

            // Saves bonus file, and notifies players.
            gameRoom.computeBonus({amt: true, addDisconnected: true, append: true, backup: false});

            // Dump all memory.
            // memory.save('memory_all.json');

            // Save times of all stages.
            memory.done.save('times.csv', {
                header: [
                    'session', 'player', 'stage', 'step', 'round', 'timestamp',
                    'time', 'timeup'
                ]
            });
            // Save data to csv
            memory.select('recordType', '=', 'decision').save('data.csv', {
                header: [
                    'session', 'group', 'player', 'WorkerId', 'mturkid','type',
                    'stage', 'step', 'round', 'time', 'timestamp','phase','trial',
                    'email_id', 'email_type','class_val','classification',
                    'class_time','confidence','conf_time','accuracy'
                ]
            });

            this.pl.each(function(p) {
                let player = channel.registry.getClient(p.id);

                //modify group id for all stages to match group id of Phase 2
                for(let i = 0; i < player.data.length; i++) {
                    player.data[i].group = player.group;
                };

                //write player.data to a data.csv file (if exists, append)
                if (fs.existsSync('./games_available/Personalized_Phishing_Training_random_v1/data/data.csv')) {
                    let output = stringify.stringify(player.data,{header: false});
                    try {
                        fs.appendFileSync('./games_available/Personalized_Phishing_Training_random_v1/data/data.csv', output, 'utf8');
                        console.log('Data saved for player '+p.id);
                    } catch (err) {
                        console.log('Some error occured - file either not saved or corrupted file saved for player '+p.id);
                    };
                } else {
                    let output = stringify.stringify(player.data,{header: true});
                    try {
                        fs.writeFileSync('./games_available/Personalized_Phishing_Training_random_v1/data/data.csv', output);
                        console.log('Data saved for player '+p.id);
                    } catch (err) {
                        console.log('Some error occured - file either not saved or corrupted file saved for player '+p.id);
                    };
                };

                let totalTime = Date.now()-player.startTime;
                let bonus = [{
                    id: player.id,
                    type: player.clientType,
                    workerid: player.WorkerId,
                    hitid: player.HITId,
                    assignmentid: player.AssignmentId,
                    access: 'NA',
                    exit: player.ExitCode,
                    totaltime: totalTime,
                    approve: 1,
                    reject: 0,
                    basepay: node.game.settings.BASE_PAY,
                    bonus: player.winRaw*node.game.settings.EXCHANGE_RATE,
                    totalpay: node.game.settings.BASE_PAY+(player.winRaw*node.game.settings.EXCHANGE_RATE), //basepay+bonus
                    disconnected: player.disconnected == null ? 0 : player.disconnected,
                    disconnectStage: 'NA'
                }];

                // Save bonus info to main bonus.csv
                if (fs.existsSync('./games_available/Personalized_Phishing_Training_random_v1/data/bonus.csv')) {
                    let output = stringify.stringify(bonus,{header: false});
                    try {
                        fs.appendFileSync('./games_available/Personalized_Phishing_Training_random_v1/data/bonus.csv', output, 'utf8');
                        console.log('Bonus data saved for player '+p.id);
                    } catch (err) {
                        console.log('Some error occured - file either not saved or corrupted file saved for player '+p.id);
                    };
                } else {
                    let output = stringify.stringify(bonus,{header: true});
                    try {
                        fs.writeFileSync('./games_available/Personalized_Phishing_Training_random_v1/data/bonus.csv', output, 'utf8');
                        console.log('Bonus data saved for player '+p.id);
                    } catch (err) {
                        console.log('Some error occured - file either not saved or corrupted file saved for player '+p.id);
                    };
                };
            });

            //node.game.gameover();
        }
    });

    stager.setOnGameOver(function() {
        // Something to do. Need to close out all open streams?
    });
};
