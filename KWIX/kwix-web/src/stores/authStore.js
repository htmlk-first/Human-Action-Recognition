import { makeAutoObservable } from "mobx";

export class AuthStore {
	isAuth = false;
	constructor() {
		makeAutoObservable(this);
	}

	login(email) {
		localStorage.setItem("kwixUser", email);
		this.isAuth = true;
	}

	logout() {
		localStorage.removeItem("kwixUser");
		this.isAuth = false;
	}
}
