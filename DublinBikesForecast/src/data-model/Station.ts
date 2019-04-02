export class Station {
    number: number;
    name: string;
    address: string;
    bikeStands?: number;
    availableStands?: number;
    availableBikes?: number;
    lastUpdate?: string;
    position?: string;
    lat?: number;
    lng?: number;
    open: boolean;

    public constructor(number: number, name: string, address: string) {
        this.number = number;
        this.name = name;
        this.address = address;
    }
}