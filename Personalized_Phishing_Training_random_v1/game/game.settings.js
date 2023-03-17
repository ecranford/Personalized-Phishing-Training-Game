/**
 * # Game settings definition file
 * Copyright(c) 2022 Edward Cranford <cranford@cmu.edu>
 * MIT Licensed
 *
 * The variables in this file will be sent to each client and saved under:
 *
 *   `node.game.settings`
 *
 * The name of the chosen treatment will be added as:
 *
 *    `node.game.settings.treatmentName`
 *
 * http://www.nodegame.org
 * ---
 */
module.exports = {

    // Variables shared by all treatments.

    // #nodeGame properties:

    /**
     * ### TIMER (object) [nodegame-property]
     *
     * Maps the names of the steps of the game to timer durations
     *
     * If a step name is found here, then the value of the property is
     * used to initialize the game timer for the step.
     */
    TIMER: {
        'terms and conditions': 45000,
        'consent form': 90000,
        demographics: 60000,
        'quick survey': 90000,
        instructions: 90000,
        'instruction quiz': 60000,
        'practice instructions': 30000,
        'practice trials': 60000,
        'start game': 30000,
        'phase 1': 30000,
        'phase 1 feedback': 10000,
        'phase 2': 30000,
        'feedback': 10000,
        'phase 2 feedback': 10000,
        'phase 3': 30000,
        'phase 3 feedback': 10000
    },

    // # Game specific properties

    // Base Pay
    BASE_PAY: 4,
    // Exchange rate coins to dollars.
    EXCHANGE_RATE: 0.05

    // # Treatments definition.

    // They can contain any number of properties, and also overwrite
    // those defined above.

    // If the `treatments` object is missing a treatment named _standard_
    // will be created automatically, and will contain all variables.

    // only 1 treatment group for PPT so comment out and single default "_standard_" will be created
    //treatments: {

    //    standard: {
    //        description: "Longer time",
    //        bidTime: 30000
    //    },

    //    pressure: {
    //        description: "Short times to take decisions",
    //        bidTime: 10000
    //    }

    //}
};
