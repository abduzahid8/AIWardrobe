/**
 * Jest config for the backend API (ESM — run with --experimental-vm-modules).
 */
module.exports = {
    testEnvironment: "node",
    testMatch: ["<rootDir>/tests/**/*.test.js"],
    verbose: true,
};
