import axios from "axios";

const api = axios.create();

api.defaults.baseURL =
	process.env.NODE_ENV === "production"
		? process.env.REACT_APP_SERVER_URL
		: "http://localhost:5000";

/**
 *
 * @param {string} email
 * @returns
 */
export const recommend = (email) => api.post("/recommend", { email });

export const auth = {
	login: (form) => api.post("/login", form),
	logout: () => api.get("/logout"),
};

export const user = {
	register: (form) => api.post("/sign_up", form),
	get: (email) => api.post("/user", { email }),
	edit: (user) => api.post("/input", user),
};
